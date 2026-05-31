document.addEventListener('DOMContentLoaded', () => {
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');

    // Handle suggestion chips
    const chips = document.querySelectorAll('.chip');
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            promptInput.value = chip.textContent;
            sendBtn.disabled = false;
            // Optionally, we can auto-send when clicked:
            // sendMessage();
        });
    });

    // Auto-resize textarea
    promptInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value.trim() === '') {
            sendBtn.disabled = true;
        } else {
            sendBtn.disabled = false;
        }
    });

    sendBtn.disabled = true;

    // Handle Enter key (Shift+Enter for new line)
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
        const text = promptInput.value.trim();
        if (!text) return;

        // Reset input
        promptInput.value = '';
        promptInput.style.height = 'auto';
        sendBtn.disabled = true;

        // Add user message
        addMessage(text, 'user');

        // Add typing indicator
        const typingId = addTypingIndicator();

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: text,
                    max_new_tokens: 150,
                    temperature: 0.8
                })
            });

            removeElement(typingId);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error');
            }

            const messageId = addMessage('', 'system');
            const messageElement = document.getElementById(messageId).querySelector('.message-content');
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        try {
                            const data = JSON.parse(dataStr);
                            messageElement.textContent += data.text;
                        } catch (e) {
                            console.error('Error parsing JSON from stream', e);
                        }
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                }
            }
        } catch (error) {
            removeElement(typingId);
            addMessage('Error: ' + error.message, 'system');
        }
    }

    function addMessage(text, sender) {
        const id = 'msg-' + Date.now() + '-' + Math.floor(Math.random() * 1000);
        const msgDiv = document.createElement('div');
        msgDiv.id = id;
        msgDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        msgDiv.appendChild(contentDiv);
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const indicator = document.createElement('div');
        indicator.id = id;
        indicator.className = 'typing-indicator';
        indicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        chatContainer.appendChild(indicator);
        scrollToBottom();
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) {
            el.remove();
        }
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
