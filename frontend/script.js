const API_BASE = "http://127.0.0.1:8000";

const pages = ["dashboard", "papers", "assistant", "library"];

const pageNames = {
    dashboard: "Overview",
    papers: "Explore papers",
    assistant: "AI research assistant",
    library: "My library"
};

const toast = document.getElementById("toast");

let toastTimer;
let savedPapers = [];

function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function navigate(page) {
    if (!pages.includes(page)) return;

    document.querySelectorAll(".page").forEach(section => {
        section.classList.toggle("active", section.id === `page-${page}`);
    });

    document.querySelectorAll(".nav-item").forEach(button => {
        button.classList.toggle("active", button.dataset.page === page);
    });

    document.getElementById("breadcrumbPage").textContent = pageNames[page];
    document.getElementById("sidebar").classList.remove("open");

    if (page === "library") renderLibrary();
}

document.addEventListener("click", event => {
    const navButton = event.target.closest("[data-page]");

    if (navButton) {
        event.preventDefault();
        navigate(navButton.dataset.page);
        return;
    }

    const queryButton = event.target.closest("[data-query]");

    if (queryButton) {
        const query = queryButton.dataset.query;
        document.getElementById("paperQuery").value = query;
        navigate("papers");
        searchPapers(query);
        return;
    }

    const chatButton = event.target.closest("[data-chat]");

    if (chatButton) {
        document.getElementById("chatInput").value = chatButton.dataset.chat;
        navigate("assistant");
        document.getElementById("chatInput").focus();
    }
});

document.getElementById("mobileMenu").addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("open");
});

document.getElementById("heroSearch").addEventListener("submit", event => {
    event.preventDefault();

    const query = document.getElementById("heroQuery").value.trim();

    if (!query) {
        showToast("Please enter a research topic.");
        return;
    }

    document.getElementById("paperQuery").value = query;
    navigate("papers");
    searchPapers(query);
});

document.getElementById("paperSearch").addEventListener("submit", event => {
    event.preventDefault();

    const query = document.getElementById("paperQuery").value.trim();

    if (!query) {
        showToast("Enter a research topic first.");
        return;
    }

    searchPapers(query);
});

async function searchPapers(query) {
    const results = document.getElementById("paperResults");
    results.replaceChildren();

    const loading = document.createElement("div");
    loading.className = "empty-state";
    loading.textContent = "Searching research papers...";
    results.appendChild(loading);

    try {
        const response = await fetch(`${API_BASE}/api/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, top_k: 5 })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Paper search failed.");
        }

        results.replaceChildren();

        if (!data.results || data.results.length === 0) {
            showEmptyState(results, "No papers found", "Try another research topic.");
            return;
        }

        data.results.forEach(paper => {
            const card = document.createElement("article");
            card.className = "paper-result";

            const title = document.createElement("h3");
            title.textContent = paper.title || "Untitled paper";

            const authors = document.createElement("p");
            authors.textContent = paper.authors || "Authors not available";

            const abstract = document.createElement("p");
            abstract.textContent = paper.abstract || "Abstract not available";

            const actions = document.createElement("div");
            actions.className = "paper-actions";

            if (isSafeHttpUrl(paper.url)) {
                const link = document.createElement("a");
                link.href = paper.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = "Read paper ↗";
                actions.appendChild(link);
            }

            const saveButton = document.createElement("button");
            saveButton.type = "button";
            saveButton.className = "secondary-button";
            saveButton.textContent = "Save paper";
            saveButton.addEventListener("click", () => savePaper(paper));
            actions.appendChild(saveButton);

            card.append(title, authors, abstract, actions);
            results.appendChild(card);
        });

        showToast(`Found ${data.count} research papers.`);
    } catch (error) {
        console.error("Paper search error:", error);
        results.replaceChildren();

        showEmptyState(
            results,
            "Could not load papers",
            "Check that the FastAPI server is running on port 8000, then try again."
        );

        showToast("Search failed. Check the backend server.");
    }
}

document.getElementById("chatForm").addEventListener("submit", async event => {
    event.preventDefault();

    const input = document.getElementById("chatInput");
    const question = input.value.trim();

    if (!question) {
        showToast("Enter a research question first.");
        return;
    }

    const messages = document.getElementById("chatMessages");

    const userMessage = document.createElement("div");
    userMessage.className = "user-message";
    userMessage.textContent = question;
    messages.appendChild(userMessage);

    input.value = "";
    input.disabled = true;

    const responseBox = document.createElement("div");
    responseBox.className = "assistant-message";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "✳";

    const content = document.createElement("div");
    content.className = "message-content";

    const heading = document.createElement("strong");
    heading.textContent = "PaperPilot AI";

    const answer = document.createElement("p");
    answer.textContent = "Searching papers and generating an answer...";
    answer.style.whiteSpace = "pre-wrap";

    content.append(heading, answer);
    responseBox.append(avatar, content);
    messages.appendChild(responseBox);
    messages.scrollTop = messages.scrollHeight;

    try {
        const response = await fetch(`${API_BASE}/api/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: question })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not generate an answer.");
        }

        answer.textContent = data.answer || "No answer was returned.";

        if (data.sources && data.sources.length > 0) {
            const sourceHeading = document.createElement("strong");
            sourceHeading.textContent = "\n\nSources";

            const sourceList = document.createElement("ul");

            data.sources.forEach(source => {
                const item = document.createElement("li");

                if (isSafeHttpUrl(source.url)) {
                    const link = document.createElement("a");
                    link.href = source.url;
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                    link.textContent = source.title || "Research paper";
                    item.appendChild(link);
                } else {
                    item.textContent = source.title || "Research paper";
                }

                sourceList.appendChild(item);
            });

            content.append(sourceHeading, sourceList);
        }
    } catch (error) {
        console.error("Assistant error:", error);
        answer.textContent =
            "Sorry, the AI assistant could not respond. Check the backend server and your Gemini API configuration.";
        showToast("AI request failed. Please try again.");
    } finally {
        input.disabled = false;
        input.focus();
        messages.scrollTop = messages.scrollHeight;
    }
});

function isSafeHttpUrl(value) {
    try {
        const url = new URL(value);
        return url.protocol === "http:" || url.protocol === "https:";
    } catch {
        return false;
    }
}

function showEmptyState(container, headingText, messageText) {
    const empty = document.createElement("div");
    empty.className = "empty-state";

    const heading = document.createElement("h3");
    heading.textContent = headingText;

    const message = document.createElement("p");
    message.textContent = messageText;

    empty.append(heading, message);
    container.appendChild(empty);
}

function renderLibrary() {
    const container = document.getElementById("libraryResults");
    container.replaceChildren();

    if (!savedPapers.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";

        const heading = document.createElement("h3");
        heading.textContent = "Your library is ready";

        const paragraph = document.createElement("p");
        paragraph.textContent = "Save papers from search results to collect them here.";

        const button = document.createElement("button");
        button.className = "primary-button";
        button.textContent = "Explore papers →";
        button.addEventListener("click", () => navigate("papers"));

        empty.append(heading, paragraph, button);
        container.appendChild(empty);
        return;
    }

    savedPapers.forEach(paper => {
        const card = document.createElement("article");
        card.className = "paper-result";

        const heading = document.createElement("h3");
        heading.textContent = paper.title || "Untitled paper";

        const authors = document.createElement("p");
        authors.textContent = paper.authors || "Authors not available";

        const abstract = document.createElement("p");
        abstract.textContent = paper.abstract || "Abstract not available";

        card.append(heading, authors, abstract);

        if (isSafeHttpUrl(paper.url)) {
            const link = document.createElement("a");
            link.href = paper.url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = "Open original paper ↗";
            card.appendChild(link);
        }

        container.appendChild(card);
    });
}

function savePaper(paper) {
    if (savedPapers.some(item => item.url === paper.url)) {
        showToast("This paper is already in your library.");
        return;
    }

    savedPapers.push(paper);
    showToast("Paper saved to your library.");
}

window.paperPilot = { navigate, savePaper };

navigate("dashboard");