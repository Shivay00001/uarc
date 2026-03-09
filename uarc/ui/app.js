document.addEventListener('DOMContentLoaded', () => {

    // UI Elements
    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const backendStatus = document.getElementById('backend-status');
    const activeModel = document.getElementById('active-model');
    const clearCacheBtn = document.getElementById('clear-cache-btn');

    // Stats Elements
    const statTps = document.getElementById('stat-tps');
    const statLatency = document.getElementById('stat-latency');
    const statSaved = document.getElementById('stat-saved');
    const statRoute = document.getElementById('stat-route');

    const vramFill = document.getElementById('vram-fill');
    const vramText = document.getElementById('vram-text');
    const ramFill = document.getElementById('ram-fill');
    const ramText = document.getElementById('ram-text');

    const metricSwaps = document.getElementById('metric-swaps');
    const metricKv = document.getElementById('metric-kv');

    const cacheDonut = document.getElementById('cache-donut');
    const cachePct = document.getElementById('cache-pct');
    const cacheHits = document.getElementById('cache-hits');
    const cacheMisses = document.getElementById('cache-misses');
    const cacheSize = document.getElementById('cache-size');
    const cacheThresh = document.getElementById('cache-thresh');

    let isGenerating = false;
    let messageHistory = [];

    // Auto-resize textarea
    promptInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') this.style.height = '48px';
    });

    // Send on Enter (Shift+Enter for newline)
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    clearCacheBtn.addEventListener('click', async () => {
        try {
            await fetch('/admin/cache/clear', { method: 'POST' });
            updateStatus();
        } catch (e) { }
    });

    function appendMessage(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;

        let initial = role === 'user' ? 'Me' : 'U';

        // Simple markdown parsing for code blocks
        let formattedText = text;
        if (text.includes('```')) {
            formattedText = text.replace(/```([\w]*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        }

        msgDiv.innerHTML = `
            <div class="avatar">${initial}</div>
            <div class="bubble">${formattedText}</div>
        `;

        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msgDiv.querySelector('.bubble');
    }

    async function sendMessage() {
        if (isGenerating) return;
        const text = promptInput.value.trim();
        if (!text) return;

        promptInput.value = '';
        promptInput.style.height = '48px';

        appendMessage('user', text);
        messageHistory.push({ role: 'user', content: text });

        const assistantBubble = appendMessage('assistant', '<span class="pulse">...</span>');
        isGenerating = true;

        try {
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: "auto",
                    messages: messageHistory,
                    stream: true,
                    max_tokens: 512
                })
            });

            if (!response.ok) throw new Error('Network error');

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let fullText = '';
            assistantBubble.innerHTML = '';

            let uarcMeta = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\\n');

                for (let line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6).trim();
                        if (dataStr === '[DONE]') continue;

                        try {
                            const data = JSON.parse(dataStr);

                            // Capture model name and meta
                            if (data.model) activeModel.textContent = `Model: ${data.model}`;
                            if (data.uarc_metadata) uarcMeta = data.uarc_metadata;

                            const delta = data.choices[0].delta.content || '';
                            fullText += delta;

                            // Live markdown render (very basic)
                            let renderText = fullText;
                            if (renderText.includes('```')) {
                                renderText = renderText.replace(/```([\\w]*)\\n([\\s\\S]*?)(```|$)/g, '<pre><code>$2</code></pre>');
                            }
                            assistantBubble.innerHTML = renderText;
                            chatContainer.scrollTop = chatContainer.scrollHeight;

                            // If we got metadata mid-stream, update the UI cards
                            if (uarcMeta) {
                                updateMetaUI(uarcMeta);
                            }

                        } catch (e) { }
                    }
                }
            }

            messageHistory.push({ role: 'assistant', content: fullText });

            // Force a status update to refresh memory charts
            updateStatus();

        } catch (error) {
            assistantBubble.innerHTML = `<span style="color:var(--color-red)">Error: Connection failed. Is engine running?</span>`;
        } finally {
            isGenerating = false;
        }
    }

    function updateMetaUI(meta) {
        statTps.innerHTML = `${meta.tokens_per_second.toFixed(1)} <small>tok/s</small>`;
        statLatency.innerHTML = `${Math.round(meta.latency_ms)} <small>ms</small>`;
        statSaved.innerHTML = `${Math.round(meta.compute_saved_pct)} <small>%</small>`;
        statRoute.textContent = meta.route.toUpperCase();

        statRoute.className = 'value'; // reset
        if (meta.route === 'cache') statRoute.classList.add('text-green');
        else if (meta.route === 'draft') statRoute.classList.add('text-blue');
        else statRoute.classList.add('highlight');
    }

    async function updateStatus() {
        try {
            const res = await fetch('/status');
            if (!res.ok) throw new Error();
            const data = await res.json();

            backendStatus.textContent = "Engine Connected & Stable";
            document.querySelector('.status-indicator .dot').classList.add('pulse');

            // --- AI-VM Memory ---
            if (data.memory) {
                const vm = data.memory;
                // VRAM
                const vramPct = (vm.VRAM.used_mb / vm.VRAM.total_mb) * 100;
                vramFill.style.width = `${Math.min(100, vramPct)}%`;
                vramText.textContent = `${Math.round(vm.VRAM.used_mb)} / ${Math.round(vm.VRAM.total_mb)} MB`;

                // RAM
                const ramPct = (vm.RAM.used_mb / vm.RAM.total_mb) * 100;
                ramFill.style.width = `${Math.min(100, ramPct)}%`;
                ramText.textContent = `${Math.round(vm.RAM.used_mb)} / ${Math.round(vm.RAM.total_mb)} MB`;
            }

            // --- Modules ---
            if (data.modules) {
                // NSC
                const nsc = data.modules.nsc;
                if (nsc) {
                    cacheHits.textContent = nsc.hits;
                    cacheMisses.textContent = nsc.misses;
                    cacheSize.textContent = nsc.size;
                    cacheThresh.textContent = nsc.threshold.toFixed(2);

                    const total = nsc.hits + nsc.misses;
                    let pct = 0;
                    if (total > 0) pct = Math.round((nsc.hits / total) * 100);

                    cachePct.textContent = `${pct}%`;
                    cacheDonut.setAttribute('stroke-dasharray', `${pct}, 100`);
                }

                // AI-VM tracking
                const aivm = data.modules.aivm;
                if (aivm) {
                    metricSwaps.textContent = aivm.promotions + aivm.evictions;
                }

                // ACS tracking
                const acs = data.modules.acs;
                if (acs) {
                    metricKv.textContent = acs.kv_sharing_pairs || 0;
                }
            }

            if (data.model) {
                activeModel.textContent = `Model: ${data.model}`;
            }

        } catch (e) {
            backendStatus.textContent = "Engine Disconnected";
            document.querySelector('.status-indicator .dot').classList.remove('pulse');
        }
    }

    // Initial load
    updateStatus();
    // Poll status every 2 seconds
    setInterval(updateStatus, 2000);
});
