(function () {
    const H3_PRESET_KEYS = ["quick", "recommended", "final"];

    function isH3StudioActive() {
        return get_uiCurrentTabContent()?.id === "tab_minimax_h3_studio";
    }

    function syncH3StudioChrome() {
        const app = gradioApp();
        const active = isH3StudioActive();
        app.classList.toggle("h3-studio-active", active);

        const studio = app.querySelector("#h3-studio");
        const generate = app.querySelector("#h3-generate button, button#h3-generate");
        const cancel = app.querySelector("#h3-cancel button, button#h3-cancel");
        const busy = Boolean(generate?.disabled && generate.textContent.includes("生成"));
        if (studio) studio.setAttribute("aria-busy", String(busy));
        if (generate) {
            generate.title = "映像＋音声を生成 (Ctrl+Enter)";
            generate.setAttribute("aria-keyshortcuts", "Control+Enter Meta+Enter");
            generate.setAttribute("aria-busy", String(busy));
        }
        if (cancel) {
            cancel.title = "実行中の生成を停止 (Esc)";
            cancel.setAttribute("aria-keyshortcuts", "Escape");
        }
    }

    function getH3ModeInputs() {
        return Array.from(gradioApp().querySelectorAll("#h3-mode input[type='radio']"));
    }

    function syncH3ModeKeyboard() {
        const inputs = getH3ModeInputs();
        if (!inputs.length) return;
        const selected = inputs.find((input) => input.checked) ?? inputs[0];
        for (const input of inputs) {
            input.name = "h3-generation-mode";
            input.tabIndex = input === selected ? 0 : -1;
        }
    }

    function syncH3Mode() {
        const studio = gradioApp().querySelector("#h3-studio");
        const selected = studio?.querySelector("#h3-mode input:checked");
        if (studio && selected) studio.dataset.h3Mode = selected.value;
        syncH3ModeKeyboard();
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

    function syncH3Validation() {
        const app = gradioApp();
        const studio = app.querySelector("#h3-studio");
        const validation = app.querySelector("#h3-input-validation [data-h3-invalid]");
        const candidates = studio?.querySelectorAll("[aria-describedby='h3-input-validation-message']") ?? [];
        for (const candidate of candidates) {
            candidate.removeAttribute("aria-invalid");
            candidate.removeAttribute("aria-describedby");
        }
        if (!studio || !validation) {
            if (studio) delete studio.dataset.h3Validation;
            return;
        }

        const target = validation.dataset.h3Invalid ?? "settings";
        const selectors = {
            prompt: "#h3-prompt textarea",
            keyframes: "#h3-first-frame input[type='file'], #h3-first-frame button",
            references: "#h3-reference-images input[type='file'], #h3-reference-images button",
            settings: "#h3-aspect input, #h3-aspect button",
        };
        const control = studio.querySelector(selectors[target] ?? selectors.settings);
        if (control) {
            control.setAttribute("aria-invalid", "true");
            control.setAttribute("aria-describedby", "h3-input-validation-message");
        }

        const signature = `${target}:${validation.textContent.trim()}`;
        if (studio.dataset.h3Validation === signature) return;
        studio.dataset.h3Validation = signature;
        if (!isH3StudioActive()) return;
        requestAnimationFrame(function () {
            control?.focus({ preventScroll: true });
            validation.scrollIntoView({
                block: "center",
                behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
                    ? "auto"
                    : "smooth",
            });
        });
    }

    function syncH3DynamicState() {
        syncH3StudioChrome();
        syncH3Mode();
        syncH3PresetState();
        syncH3Validation();
    }

    function setupH3Studio() {
        syncH3StudioChrome();
        syncH3Mode();
        syncH3PresetState();
        syncH3Validation();
        const app = gradioApp();
        if (app.dataset.h3ModeListener === "ready") return;
        app.dataset.h3ModeListener = "ready";
        app.addEventListener("change", function (event) {
            if (event.target.closest?.("#h3-mode")) syncH3Mode();
        });
        app.addEventListener("keydown", function (event) {
            const input = event.target.closest?.("#h3-mode input[type='radio']");
            if (!input) return;
            const inputs = getH3ModeInputs();
            const index = inputs.indexOf(input);
            if (index < 0) return;
            let nextIndex = null;
            if (event.key === "ArrowRight" || event.key === "ArrowDown") {
                nextIndex = (index + 1) % inputs.length;
            } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
                nextIndex = (index - 1 + inputs.length) % inputs.length;
            } else if (event.key === "Home") {
                nextIndex = 0;
            } else if (event.key === "End") {
                nextIndex = inputs.length - 1;
            }
            if (nextIndex === null) return;
            event.preventDefault();
            inputs[nextIndex].click();
            inputs[nextIndex].focus();
        });
    }

    onUiLoaded(setupH3Studio);
    onUiTabChange(function () {
        syncH3DynamicState();
    });
    onAfterUiUpdate(syncH3DynamicState);

    document.addEventListener("keydown", function (event) {
        if (!isH3StudioActive() || event.isComposing || event.keyCode === 229) return;

        const app = gradioApp();
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            const generate = app.querySelector("#h3-generate button, button#h3-generate");
            if (generate && !generate.disabled && generate.getAttribute("aria-busy") !== "true") {
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
