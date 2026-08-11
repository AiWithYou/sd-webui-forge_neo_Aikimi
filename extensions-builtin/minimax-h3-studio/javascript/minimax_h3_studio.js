(function () {
    const H3_PRESET_KEYS = ["quick", "recommended", "final"];

    function isH3StudioActive() {
        return get_uiCurrentTabContent()?.id === "tab_minimax_h3_studio";
    }

    function syncH3StudioChrome() {
        const app = gradioApp();
        const active = isH3StudioActive();
        app.classList.toggle("h3-studio-active", active);

        const generate = app.querySelector("#h3-generate button, button#h3-generate");
        const cancel = app.querySelector("#h3-cancel button, button#h3-cancel");
        if (generate) generate.title = "映像＋音声を生成 (Ctrl+Enter)";
        if (cancel) cancel.title = "実行中の生成を停止 (Esc)";
    }

    function syncH3Mode() {
        const studio = gradioApp().querySelector("#h3-studio");
        const selected = studio?.querySelector("#h3-mode input:checked");
        if (studio && selected) studio.dataset.h3Mode = selected.value;
    }

    function syncH3PresetState() {
        const app = gradioApp();
        const state = app.querySelector("#h3-preset-state [data-h3-preset]");
        const selected = state?.dataset.h3Preset ?? "custom";
        for (const key of H3_PRESET_KEYS) {
            const button = app.querySelector(
                `#h3-preset-${key} button, button#h3-preset-${key}`
            );
            if (button) button.setAttribute("aria-pressed", String(selected === key));
        }
    }

    function setupH3Studio() {
        syncH3StudioChrome();
        syncH3Mode();
        syncH3PresetState();
        const app = gradioApp();
        if (app.dataset.h3ModeListener === "ready") return;
        app.dataset.h3ModeListener = "ready";
        app.addEventListener("change", function (event) {
            if (event.target.closest?.("#h3-mode")) syncH3Mode();
        });
    }

    onUiLoaded(setupH3Studio);
    onUiTabChange(function () {
        syncH3StudioChrome();
        syncH3Mode();
        syncH3PresetState();
    });
    onAfterUiUpdate(syncH3PresetState);

    document.addEventListener("keydown", function (event) {
        if (!isH3StudioActive() || event.isComposing || event.keyCode === 229) return;

        const app = gradioApp();
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            const generate = app.querySelector("#h3-generate button, button#h3-generate");
            if (generate && !generate.disabled) {
                event.preventDefault();
                generate.click();
            }
        } else if (event.key === "Escape") {
            const cancel = app.querySelector("#h3-cancel button, button#h3-cancel");
            if (cancel && !cancel.disabled) {
                event.preventDefault();
                cancel.click();
            }
        }
    });
})();
