(function () {
    "use strict";

    const API_VERSION = 3;
    const FEATURE_EVENT = "aikimi:feature-tab-change";
    const FEATURE_NAV_ID = "aikimi-feature-nav";
    const KREA2_SCRIPT = "Krea2 2-Stage Upscale";
    const SETUP_RETRY_LIMIT = 80;
    const SETUP_RETRY_DELAY_MS = 50;
    const REPAIR_LIMIT = 12;

    const FEATURES = Object.freeze({
        krea2: Object.freeze({
            buttonId: "aikimi-tab-krea2",
            containerId: "tab_img2img",
            label: "Krea2",
            kind: "alias",
        }),
        anima38: Object.freeze({
            buttonId: "aikimi-tab-anima38",
            containerId: "tab_txt2img",
            label: "Anima",
            kind: "alias",
        }),
        sensenova: Object.freeze({
            buttonId: "aikimi-tab-sensenova",
            containerId: "tab_sensenova_u15_studio",
            label: "SenseNova",
            kind: "native",
        }),
        minimax_h3: Object.freeze({
            buttonId: "aikimi-tab-minimax-h3",
            containerId: "tab_minimax_h3_studio",
            label: "MiniMax H3",
            kind: "native",
        }),
    });
    const FEATURE_ORDER = Object.freeze([
        "krea2",
        "anima38",
        "sensenova",
        "minimax_h3",
    ]);

    if (window.AikimiTabs?.apiVersion === API_VERSION) {
        window.AikimiTabs.refresh();
        return;
    }

    let activeFeature = null;
    let activatingFeature = null;
    let activationSequence = 0;
    let setupTimer = null;
    let reconcileTimer = null;
    let repairCount = 0;

    function appRoot() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function topTabs() {
        return appRoot().querySelector("#tabs");
    }

    function tabListFor(tabs) {
        if (!tabs) return null;
        if (typeof get_uiTabList === "function") return get_uiTabList(tabs);
        return tabs.querySelector(
            ':scope > .tab-wrapper > .tab-container[role="tablist"], :scope > .tab-nav[role="tablist"], :scope > .tab-nav',
        );
    }

    function topTabNav() {
        return tabListFor(topTabs());
    }

    function nativeTabButtons() {
        const nav = topTabNav();
        if (!nav) return [];
        return Array.from(nav.children).filter(function (element) {
            return element.tagName === "BUTTON";
        });
    }

    function topTabItems() {
        const tabs = topTabs();
        if (!tabs) return [];
        return Array.from(tabs.children).filter(function (element) {
            return element.classList?.contains("tabitem") && element.id?.startsWith("tab_");
        });
    }

    function nativeButtonFor(containerId) {
        if (typeof get_uiTopTabButton === "function") {
            const helperButton = get_uiTopTabButton(containerId);
            if (helperButton?.parentElement === topTabNav()) return helperButton;
        }

        const buttons = nativeTabButtons();
        const controlled = buttons.find(function (button) {
            return button.getAttribute("aria-controls") === containerId;
        });
        if (controlled) return controlled;

        const identified = buttons.find(function (button) {
            return button.id === `${containerId}-button`;
        });
        if (identified) return identified;

        const panelIndex = topTabItems().findIndex(function (panel) {
            return panel.id === containerId;
        });
        return panelIndex >= 0 ? buttons[panelIndex] || null : null;
    }

    function featureContainer(feature) {
        const config = FEATURES[feature];
        return config ? appRoot().querySelector(`#${config.containerId}`) : null;
    }

    function featureNavigation() {
        return appRoot().querySelector(`#${FEATURE_NAV_ID}`);
    }

    function featureButton(feature) {
        const config = FEATURES[feature];
        return config ? featureNavigation()?.querySelector(`#${config.buttonId}`) : null;
    }

    function dispatchFeatureChange(feature, warning) {
        const container = feature ? featureContainer(feature) : null;
        document.dispatchEvent(
            new CustomEvent(FEATURE_EVENT, {
                detail: Object.freeze({
                    feature,
                    container,
                    ready: !warning,
                    warning: warning || null,
                }),
            }),
        );
    }

    function syncFeatureButtonState(feature) {
        const row = featureNavigation();
        if (!row) return;
        if (feature) {
            row.dataset.activeFeature = feature;
        } else {
            delete row.dataset.activeFeature;
        }
        FEATURE_ORDER.forEach(function (candidate) {
            const button = featureButton(candidate);
            if (!button) return;
            const isActive = candidate === feature;
            button.classList.toggle("aikimi-feature-active", isActive);
            if (isActive) {
                button.setAttribute("aria-current", "page");
            } else {
                button.removeAttribute("aria-current");
            }
        });
    }

    function setActiveFeature(feature, warning) {
        activeFeature = feature && FEATURES[feature] ? feature : null;
        syncFeatureButtonState(activeFeature);
        dispatchFeatureChange(activeFeature, warning);
    }

    function createFeatureButton(feature) {
        const config = FEATURES[feature];
        const button = document.createElement("button");
        button.type = "button";
        button.id = config.buttonId;
        button.className = `aikimi-feature-nav__button aikimi-feature-nav__button--${feature}`;
        button.dataset.aikimiFeature = feature;
        button.setAttribute("aria-controls", config.containerId);
        button.setAttribute("aria-label", `${config.label}をForgeで開く`);
        button.textContent = config.label;
        return button;
    }

    function createFeatureNavigation() {
        const row = document.createElement("nav");
        row.id = FEATURE_NAV_ID;
        row.className = "aikimi-feature-nav";
        row.setAttribute("aria-label", "Aikimi機能");
        FEATURE_ORDER.forEach(function (feature) {
            row.append(createFeatureButton(feature));
        });
        row.addEventListener("click", handleFeatureNavigationClick);
        row.addEventListener("keydown", handleFeatureNavigationKeydown);
        return row;
    }

    function ensureFeatureNavigation() {
        const tabs = topTabs();
        const nativeNav = topTabNav();
        if (!tabs?.parentElement || !nativeNav) return false;

        let row = featureNavigation();
        if (!row) row = createFeatureNavigation();
        if (row.parentElement !== tabs.parentElement || row.nextElementSibling !== tabs) {
            tabs.parentElement.insertBefore(row, tabs);
        }

        let availableCount = 0;
        FEATURE_ORDER.forEach(function (feature) {
            const button = featureButton(feature);
            const available = Boolean(nativeButtonFor(FEATURES[feature].containerId));
            button.hidden = !available;
            button.disabled = !available;
            if (available) availableCount += 1;
        });
        row.hidden = availableCount === 0;
        syncFeatureButtonState(activeFeature || activatingFeature);
        return true;
    }

    function waitFor(predicate, attempts = 40, delayMs = SETUP_RETRY_DELAY_MS) {
        return new Promise(function (resolve) {
            let remaining = attempts;
            function check() {
                const value = predicate();
                if (value || remaining <= 0) {
                    resolve(value || null);
                    return;
                }
                remaining -= 1;
                window.setTimeout(check, delayMs);
            }
            check();
        });
    }

    function normalizeGradioOptionLabel(value) {
        if (typeof value !== "string") return null;
        const trimmed = value.trim();
        return trimmed.startsWith("✓") ? trimmed.slice(1).trim() : trimmed;
    }

    function exactVisibleOption(label) {
        return Array.from(appRoot().querySelectorAll("[role='option']")).find(function (option) {
            if (
                option.getAttribute("aria-disabled") === "true" ||
                option.offsetParent === null
            ) {
                return false;
            }
            const candidates = [
                option.getAttribute("aria-label"),
                option.getAttribute("data-value"),
                option.getAttribute("value"),
                option.textContent,
            ];
            return candidates.some(function (candidate) {
                return normalizeGradioOptionLabel(candidate) === label;
            });
        }) || null;
    }

    function activateDropdownOption(option) {
        const gradio6Option = option.matches("li[data-index]") &&
            Boolean(option.closest("ul[role='listbox']"));
        if (gradio6Option) {
            option.dispatchEvent(new MouseEvent("mousedown", {
                bubbles: true,
                cancelable: true,
                view: window,
            }));
        } else {
            option.click();
        }
    }

    function openGradioDropdown(input) {
        input.focus();
        input.dispatchEvent(new KeyboardEvent("keydown", {
            key: "ArrowDown",
            code: "ArrowDown",
            bubbles: true,
            cancelable: true,
        }));
    }

    async function selectNormalImg2ImgMode() {
        const button = await waitFor(function () {
            return tabListFor(appRoot().querySelector("#mode_img2img"))
                ?.querySelector(":scope > button:first-child");
        });
        if (!button) {
            return "通常のimg2imgを開けませんでした。img2imgタブから手動で選択してください。";
        }
        button.click();
        return null;
    }

    async function selectKrea2Script() {
        const dropdown = await waitFor(function () {
            return appRoot().querySelector("#img2img_script_container #script_list");
        });
        if (!dropdown) return "Krea2のScript選択欄が見つかりません。UIを再読み込みしてください。";

        const input = dropdown.querySelector("input");
        if (!input) return "Krea2のScript選択欄を操作できません。UIを再読み込みしてください。";

        const panelSelector = "#script_krea2_2stage_upscale_quick_4k";
        if (input.value !== KREA2_SCRIPT) {
            openGradioDropdown(input);
            let option = await waitFor(function () {
                return exactVisibleOption(KREA2_SCRIPT);
            }, 20);
            if (!option) {
                openGradioDropdown(input);
                option = await waitFor(function () {
                    return exactVisibleOption(KREA2_SCRIPT);
                }, 40);
            }
            if (option) activateDropdownOption(option);
            await waitFor(function () {
                return input.value === KREA2_SCRIPT ? input : null;
            }, 60);
        }

        if (input.value !== KREA2_SCRIPT) {
            return "Krea2 2-Stage Upscaleを選択できませんでした。Script欄から手動で選択してください。";
        }

        const panelControl = await waitFor(function () {
            const element = appRoot().querySelector(panelSelector);
            return element && element.offsetParent !== null ? element : null;
        }, 60);
        if (!panelControl) {
            return "Krea2 2-Stage Upscaleを選択できませんでした。Script欄から手動で選択してください。";
        }
        return null;
    }

    async function expandAnimaAccordion() {
        const accordionId = "aikimi-txt2img-anima38";
        const controls = await waitFor(function () {
            const accordion = appRoot().querySelector(`#${accordionId}`);
            const visibleCheckbox = appRoot().querySelector(
                `#${accordionId}-visible-checkbox`,
            );
            const hiddenCheckbox = appRoot().querySelector(
                `#${accordionId}-checkbox input[type='checkbox']`,
            );
            const label = accordion?.querySelector(".label-wrap");
            if (
                !accordion ||
                !label ||
                !visibleCheckbox ||
                !hiddenCheckbox ||
                !accordion.visibleCheckbox ||
                typeof accordion.onVisibleCheckboxChange !== "function" ||
                typeof inputAccordionChecked !== "function"
            ) {
                return null;
            }
            return { accordion, label, visibleCheckbox, hiddenCheckbox };
        }, 60);
        if (!controls) return "Anima 3.8Bの有効化欄が準備できません。UIを再読み込みしてください。";

        inputAccordionChecked(accordionId, true);
        const opened = await waitFor(function () {
            return controls.label.classList.contains("open") &&
                controls.visibleCheckbox.checked &&
                controls.hiddenCheckbox.checked
                ? controls.accordion
                : null;
        }, 20);
        return opened ? null : "Anima 3.8Bを有効化できませんでした。設定欄を手動で有効にしてください。";
    }

    function clickNativeTab(containerId) {
        const button = nativeButtonFor(containerId);
        if (!button) return false;
        button.click();
        return true;
    }

    async function activateFeature(feature) {
        const sequence = ++activationSequence;
        const config = FEATURES[feature];
        if (!config) return;

        activatingFeature = feature;
        syncFeatureButtonState(feature);
        if (!clickNativeTab(config.containerId)) {
            activatingFeature = null;
            setActiveFeature(null, `${config.label}のForgeタブが見つかりません。UIを再読み込みしてください。`);
            return;
        }

        let warning = null;
        if (feature === "krea2") {
            const modeWarning = await selectNormalImg2ImgMode();
            const scriptWarning = await selectKrea2Script();
            warning = modeWarning || scriptWarning;
        } else if (feature === "anima38") {
            warning = await expandAnimaAccordion();
        } else {
            const mounted = await waitFor(function () {
                return featureContainer(feature);
            }, 60);
            if (!mounted) {
                warning = `${config.label}を開けませんでした。UIを再読み込みしてください。`;
            }
        }

        if (sequence !== activationSequence) return;
        activatingFeature = null;
        setActiveFeature(feature, warning);
    }

    function selectedNativeButton() {
        return nativeTabButtons().find(function (button) {
            return button.classList.contains("selected") ||
                button.getAttribute("aria-selected") === "true";
        }) || null;
    }

    function nativeFeatureFor(button) {
        return FEATURE_ORDER.find(function (feature) {
            return FEATURES[feature].kind === "native" &&
                nativeButtonFor(FEATURES[feature].containerId) === button;
        }) || null;
    }

    function selectedButtonMatches(feature, selected) {
        const config = FEATURES[feature];
        return Boolean(config && nativeButtonFor(config.containerId) === selected);
    }

    function aliasStateMatches(feature) {
        if (feature === "krea2") {
            const modeButton = tabListFor(appRoot().querySelector("#mode_img2img"))
                ?.querySelector(":scope > button:first-child");
            const scriptInput = appRoot().querySelector("#img2img_script_container #script_list input");
            const panel = appRoot().querySelector("#script_krea2_2stage_upscale_quick_4k");
            const normalModeSelected = Boolean(
                modeButton &&
                (modeButton.classList.contains("selected") || modeButton.getAttribute("aria-selected") === "true"),
            );
            return normalModeSelected && scriptInput?.value === KREA2_SCRIPT && panel?.offsetParent !== null;
        }
        if (feature === "anima38") {
            const accordion = appRoot().querySelector("#aikimi-txt2img-anima38");
            const visibleCheckbox = appRoot().querySelector("#aikimi-txt2img-anima38-visible-checkbox");
            const hiddenCheckbox = appRoot().querySelector(
                "#aikimi-txt2img-anima38-checkbox input[type='checkbox']",
            );
            return Boolean(
                accordion?.querySelector(".label-wrap")?.classList.contains("open") &&
                visibleCheckbox?.checked &&
                hiddenCheckbox?.checked,
            );
        }
        return false;
    }

    function redispatchWhenContainerMounts(feature) {
        if (featureContainer(feature)) return;
        void waitFor(function () {
            return featureContainer(feature);
        }, 60).then(function (container) {
            if (container && activeFeature === feature) dispatchFeatureChange(feature, null);
        });
    }

    function syncFromSelectedNativeTab() {
        const selected = selectedNativeButton();
        if (!selected) return;
        if (activatingFeature && selectedButtonMatches(activatingFeature, selected)) {
            syncFeatureButtonState(activatingFeature);
            return;
        }

        const nativeFeature = nativeFeatureFor(selected);
        if (nativeFeature) {
            if (activeFeature !== nativeFeature) setActiveFeature(nativeFeature, null);
            redispatchWhenContainerMounts(nativeFeature);
            return;
        }

        if (
            activeFeature &&
            FEATURES[activeFeature]?.kind === "alias" &&
            selectedButtonMatches(activeFeature, selected)
        ) {
            if (aliasStateMatches(activeFeature)) {
                syncFeatureButtonState(activeFeature);
            } else {
                setActiveFeature(null, null);
            }
            return;
        }
        if (activeFeature !== null) setActiveFeature(null, null);
    }

    function scheduleAliasReconciliation() {
        if (
            reconcileTimer !== null ||
            activatingFeature ||
            !activeFeature ||
            FEATURES[activeFeature]?.kind !== "alias"
        ) {
            return;
        }
        reconcileTimer = window.setTimeout(function () {
            reconcileTimer = null;
            syncFromSelectedNativeTab();
        }, 0);
    }

    function setup() {
        if (!ensureFeatureNavigation()) return false;
        syncFromSelectedNativeTab();
        return true;
    }

    function setupWithRetry(attempt = 0) {
        if (setup()) return;
        if (attempt >= SETUP_RETRY_LIMIT) return;
        window.setTimeout(function () {
            setupWithRetry(attempt + 1);
        }, SETUP_RETRY_DELAY_MS);
    }

    function scheduleSetup() {
        if (setupTimer !== null) return;
        setupTimer = window.setTimeout(function () {
            setupTimer = null;
            setup();
        }, 0);
    }

    function mutationContainsNode(nodes, predicate) {
        return Array.from(nodes || []).some(function (node) {
            return node.nodeType === Node.ELEMENT_NODE && predicate(node);
        });
    }

    function repairExternalNavigationOnUiUpdate(mutationRecords) {
        const tabs = topTabs();
        const row = featureNavigation();
        if (row && tabs && row.parentElement === tabs.parentElement && row.nextElementSibling === tabs) {
            return;
        }
        const relevant = Array.from(mutationRecords || []).some(function (record) {
            if (record.target === row || row?.contains(record.target)) return false;
            return mutationContainsNode(record.removedNodes, function (node) {
                return node.id === FEATURE_NAV_ID || node.id === "tabs" ||
                    Boolean(node.querySelector?.(`#${FEATURE_NAV_ID}, #tabs`));
            }) || mutationContainsNode(record.addedNodes, function (node) {
                return node.id === "tabs" || Boolean(node.querySelector?.("#tabs"));
            });
        });
        if (!relevant || repairCount >= REPAIR_LIMIT) return;
        repairCount += 1;
        scheduleSetup();
    }

    function handleUiUpdate(mutationRecords) {
        repairExternalNavigationOnUiUpdate(mutationRecords);
        scheduleAliasReconciliation();
    }

    function topNavButtonFromTarget(target) {
        const button = target?.closest?.("button[role='tab']");
        return button?.parentElement === topTabNav() ? button : null;
    }

    function handleNativeTabClick(event) {
        const button = topNavButtonFromTarget(event.target);
        if (!button) return;
        if (activatingFeature && selectedButtonMatches(activatingFeature, button)) return;

        activationSequence += 1;
        activatingFeature = null;
        const nativeFeature = nativeFeatureFor(button);
        if (nativeFeature) {
            setActiveFeature(nativeFeature, null);
            redispatchWhenContainerMounts(nativeFeature);
        } else {
            setActiveFeature(null, null);
        }
    }

    function handleFeatureNavigationClick(event) {
        const button = event.target.closest?.(".aikimi-feature-nav__button");
        if (!button || button.parentElement !== featureNavigation() || button.disabled) return;
        void activateFeature(button.dataset.aikimiFeature);
    }

    function handleFeatureNavigationKeydown(event) {
        const button = event.target.closest?.(".aikimi-feature-nav__button");
        if (!button || button.parentElement !== featureNavigation() || event.defaultPrevented) return;
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            button.click();
            return;
        }
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;

        const buttons = Array.from(
            featureNavigation()?.querySelectorAll(":scope > .aikimi-feature-nav__button:not([hidden])") || [],
        );
        const index = buttons.indexOf(button);
        if (index < 0 || !buttons.length) return;
        let nextIndex = index;
        if (event.key === "ArrowLeft") nextIndex = (index - 1 + buttons.length) % buttons.length;
        if (event.key === "ArrowRight") nextIndex = (index + 1) % buttons.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = buttons.length - 1;
        event.preventDefault();
        buttons[nextIndex].focus();
    }

    window.AikimiTabs = Object.freeze({
        apiVersion: API_VERSION,
        getActiveFeature: function () {
            return activeFeature;
        },
        getActiveContainer: function () {
            return activeFeature ? featureContainer(activeFeature) : null;
        },
        refresh: function () {
            scheduleSetup();
        },
    });

    document.addEventListener("click", handleNativeTabClick);
    document.addEventListener("click", scheduleAliasReconciliation);
    document.addEventListener("input", scheduleAliasReconciliation);
    document.addEventListener("change", scheduleAliasReconciliation);
    onUiLoaded(function () {
        setupWithRetry();
    });
    onUiUpdate(handleUiUpdate);
    onUiTabChange(syncFromSelectedNativeTab);
})();
