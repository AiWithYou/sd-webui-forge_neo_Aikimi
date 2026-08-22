(function () {
    const DRAFT_KEY = "forge-neo:sensenova-u15:prompt-draft:v1";
    let draftTimer = null;

    function studioIsActive() {
        const studio = gradioApp().querySelector("#sensenova-u15-studio");
        return Boolean(studio && studio.offsetParent !== null);
    }

    function promptElement() {
        return gradioApp().querySelector("#sn-prompt textarea");
    }

    function buttonElement(id) {
        return gradioApp().querySelector(`#${id} button, button#${id}`);
    }

    function syncPromptCount() {
        const prompt = promptElement();
        const counter = gradioApp().querySelector("#sn-prompt-count");
        if (!prompt || !counter) return;
        const length = Array.from(prompt.value).length;
        counter.textContent = `${length.toLocaleString()} / 20,000`;
        counter.dataset.overLimit = String(length > 20000);
    }

    function saveDraft() {
        const prompt = promptElement();
        const status = gradioApp().querySelector("#sn-draft-status");
        if (!prompt) return;
        try {
            if (prompt.value) {
                window.localStorage.setItem(DRAFT_KEY, prompt.value);
                if (status) status.textContent = "下書きをこの端末に保存済み";
            } else {
                window.localStorage.removeItem(DRAFT_KEY);
                if (status) status.textContent = "下書きをこの端末に自動保存";
            }
        } catch (_error) {
            if (status) status.textContent = "下書き保存は利用できません";
        }
    }

    function scheduleDraftSave() {
        if (draftTimer !== null) window.clearTimeout(draftTimer);
        draftTimer = window.setTimeout(function () {
            draftTimer = null;
            saveDraft();
        }, 300);
    }

    function syncDraftStatus() {
        const prompt = promptElement();
        const status = gradioApp().querySelector("#sn-draft-status");
        if (!prompt || !status || !prompt.value) return;
        if (status.textContent === "下書きをこの端末に自動保存") {
            status.textContent = "前回の下書きを復元しました";
        }
    }

    function syncBusyState() {
        const progress = gradioApp().querySelector("#sn-progress [data-stage]");
        const generate = buttonElement("sn-generate");
        if (!generate || !progress) return;
        const stage = progress.dataset.stage || "idle";
        const busy = !["idle", "complete", "error", "cancelled", "cancel"].includes(stage);
        generate.setAttribute("aria-busy", String(busy));
    }

    function setupStudio() {
        const app = gradioApp();
        app.addEventListener("input", function (event) {
            if (!event.target.closest?.("#sn-prompt")) return;
            syncPromptCount();
            scheduleDraftSave();
        });
        syncPromptCount();
        syncDraftStatus();
        syncBusyState();
    }

    onUiLoaded(setupStudio);
    onAfterUiUpdate(function () {
        syncPromptCount();
        syncDraftStatus();
        syncBusyState();
    });

    document.addEventListener("keydown", function (event) {
        if (!studioIsActive() || event.defaultPrevented || event.isComposing || event.keyCode === 229) return;
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            const generate = buttonElement("sn-generate");
            if (generate && !generate.disabled && generate.getAttribute("aria-busy") !== "true") {
                event.preventDefault();
                generate.click();
            }
            return;
        }
        if (event.key !== "Escape" || event.target.closest?.("[role='dialog'], [role='listbox']")) return;
        const cancel = buttonElement("sn-cancel");
        if (cancel && !cancel.disabled) {
            event.preventDefault();
            cancel.click();
        }
    });
})();
