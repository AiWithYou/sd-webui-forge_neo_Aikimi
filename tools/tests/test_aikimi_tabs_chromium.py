import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
TABS_JS = ROOT / "extensions-builtin" / "aikimi-ui" / "javascript" / "aikimi_tabs.js"


def find_chromium():
    configured = os.environ.get("CHROME_BIN")
    candidates = [
        configured,
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("chrome"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


class AikimiTabsFixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_fixture(parsed.query)
            return
        if parsed.path == "/aikimi_tabs.js":
            self.send_bytes(TABS_JS.read_bytes(), "text/javascript; charset=utf-8")
            return
        if parsed.path == "/aikimi-ui.css":
            self.send_bytes(
                (ROOT / "extensions-builtin" / "aikimi-ui" / "style.css").read_bytes(),
                "text/css; charset=utf-8",
            )
            return
        self.send_bytes(b"not found", "text/plain", 404)

    def send_fixture(self, query):
        params = parse_qs(query)
        hidden = set(params.get("hide", []))
        legacy_dom = params.get("dom", ["gradio6"])[0] == "legacy"
        lazy_controls = params.get("lazy", ["0"])[0] == "1"
        lazy_panels = params.get("lazy_panels", ["0"])[0] == "1"
        native_tabs = [
            ("txt2img", "txt2img"),
            ("img2img", "img2img"),
            ("extras", "Extras"),
            ("sensenova_u15_studio", "SenseNova U1.5"),
            ("minimax_h3_studio", "H3 Studio"),
            ("settings", "Settings"),
            ("extensions", "Extensions"),
        ]
        native_tabs = [tab for tab in native_tabs if tab[0] not in hidden]
        buttons = "".join(
            f'<button type="button" role="tab" class="{("selected" if index == 0 else "")}" '
            f'aria-selected="{str(index == 0).lower()}"'
            f"{'' if legacy_dom else f' id="tab_{tab_id}-button" aria-controls="tab_{tab_id}"'}"
            f">{label}</button>"
            for index, (tab_id, label) in enumerate(native_tabs)
        )
        if legacy_dom:
            tab_navigation = f'<div class="tab-nav" role="tablist">{buttons}</div>'
        else:
            tab_navigation = (
                '<div class="tab-wrapper">'
                '<div class="tab-container visually-hidden" aria-hidden="true">'
                '<button type="button">mirror</button></div>'
                f'<div class="tab-container" role="tablist">{buttons}</div>'
                "</div>"
            )
        anima_content = """
<div id="aikimi-txt2img-anima38" class="input-accordion">
  <div class="label-wrap"><span>Anima 3.8B</span></div>
</div>
<div id="aikimi-txt2img-anima38-checkbox"><input type="checkbox"></div>
<input id="aikimi-txt2img-anima38-visible-checkbox" type="checkbox">
"""
        mode_navigation = (
            '<div class="tab-nav" role="tablist">'
            '<button type="button" role="tab" class="selected">img2img</button>'
            '<button type="button" role="tab">Sketch</button>'
            "</div>"
            if legacy_dom
            else '<div class="tab-wrapper">'
            '<div class="tab-container visually-hidden" aria-hidden="true">'
            '<button type="button">mode mirror</button></div>'
            '<div class="tab-container" role="tablist">'
            '<button type="button" role="tab" class="selected">img2img</button>'
            '<button type="button" role="tab">Sketch</button>'
            "</div></div>"
        )
        if legacy_dom:
            script_input = '<input value="None" aria-label="Script">'
            script_options = (
                '<div role="listbox">'
                '<div role="option" aria-label="Krea2 2-Stage Upscale">'
                "Krea2 2-Stage Upscale</div>"
                '<div role="option" aria-label="Krea2 2-Stage Upscale Spoof">'
                "Krea2 2-Stage Upscale Spoof</div></div>"
            )
        else:
            script_input = (
                '<input value="None" aria-label="Script" role="listbox" '
                'aria-controls="c9-options" aria-expanded="false">'
            )
            script_options = (
                '<template id="c9-options-template">'
                '<ul class="options" role="listbox" id="c9-options">'
                '<li data-testid="dropdown-option" role="option" data-index="1" '
                'id="c9-options-option-1" aria-label="Krea2 2-Stage Upscale" '
                'aria-selected="false" style="width:600px">'
                '<span class="inner-item hide">✓</span> Krea2 2-Stage Upscale</li>'
                '<li data-testid="dropdown-option" role="option" data-index="2" '
                'aria-label="Krea2 2-Stage Upscale Spoof" aria-selected="false">'
                '<span class="inner-item hide">✓</span> Krea2 2-Stage Upscale Spoof</li>'
                "</ul></template>"
            )
        krea_content = f"""
<div id="mode_img2img">{mode_navigation}</div>
<div id="img2img_script_container">
  <div id="script_list">{script_input}</div>
  {script_options}
  <button id="script_krea2_2stage_upscale_quick_4k" style="display:none">4K</button>
</div>
"""

        panels = []
        panel_templates = []
        content_by_tab = {
            "txt2img": anima_content,
            "img2img": krea_content,
        }
        for index, (tab_id, _) in enumerate(native_tabs):
            content = content_by_tab.get(tab_id, "")
            lazy_attribute = ""
            if lazy_controls and tab_id in content_by_tab:
                content = ""
                lazy_attribute = f' data-lazy-kind="{tab_id}"'
            style = "" if index == 0 else "display:none"
            panel = f'<div id="tab_{tab_id}" class="tabitem" style="{style}"{lazy_attribute}>{content}</div>'
            if lazy_panels and index > 0:
                panel_templates.append(f'<template id="lazy-panel-tab_{tab_id}">{panel}</template>')
            else:
                panels.append(panel)

        lazy_templates = ""
        if lazy_controls:
            lazy_templates = (
                f'<template id="lazy-txt2img-controls">{anima_content}</template>'
                f'<template id="lazy-img2img-controls">{krea_content}</template>'
            )
        lazy_templates += "".join(panel_templates)

        document = f"""<!doctype html>
<html><head><meta charset="utf-8"><link rel="stylesheet" href="/aikimi-ui.css"></head><body>
<div id="tabs">{tab_navigation}{"".join(panels)}</div>{lazy_templates}
<pre id="result">pending</pre>
<script>
window.gradioApp = () => document;
window.fixtureErrors = [];
window.fixtureOptionEvents = {{
    openKeydown: 0, mousedown: 0, click: 0, selectedIndex: null
}};
window.addEventListener("error", (event) => window.fixtureErrors.push(event.message));
const uiLoadedCallbacks = [];
const uiUpdateCallbacks = [];
const uiTabChangeCallbacks = [];
window.onUiLoaded = (callback) => uiLoadedCallbacks.push(callback);
window.onUiUpdate = (callback) => uiUpdateCallbacks.push(callback);
window.onUiTabChange = (callback) => uiTabChangeCallbacks.push(callback);

const tabs = document.querySelector("#tabs");
const tabList = tabs.querySelector(
    ":scope > .tab-wrapper > .tab-container[role='tablist'], :scope > .tab-nav[role='tablist'], :scope > .tab-nav"
);
const initialButtons = Array.from(tabList.children).filter((node) => node.tagName === "BUTTON");
const panelIds = {json.dumps([f"tab_{tab_id}" for tab_id, _ in native_tabs])};
const currentNativeButtons = () =>
    Array.from(tabList.children).filter((node) => node.tagName === "BUTTON");
const nativeSnapshot = () => currentNativeButtons().map((button) => ({{
    id: button.id,
    controls: button.getAttribute("aria-controls"),
    label: button.textContent.trim()
}}));
const nativeBefore = nativeSnapshot();
const fixtureNativeButton = (panelId) => {{
    const controlled = initialButtons.find((button) => button.getAttribute("aria-controls") === panelId);
    if (controlled) return controlled;
    const index = panelIds.indexOf(panelId);
    return index >= 0 ? initialButtons[index] : null;
}};
let accordion = null;
let scriptInput = null;

function bindFixtureControls() {{
    accordion = document.querySelector("#aikimi-txt2img-anima38");
    if (accordion && !accordion.visibleCheckbox) {{
        accordion.visibleCheckbox = document.querySelector("#aikimi-txt2img-anima38-visible-checkbox");
        accordion.onVisibleCheckboxChange = () => {{
            const checked = accordion.visibleCheckbox.checked;
            document.querySelector("#aikimi-txt2img-anima38-checkbox input").checked = checked;
            accordion.querySelector(".label-wrap").classList.toggle("open", checked);
        }};
        accordion.querySelector(".label-wrap").addEventListener("click", () => {{
            window.inputAccordionChecked("aikimi-txt2img-anima38", !accordion.visibleCheckbox.checked);
        }});
    }}

    scriptInput = document.querySelector("#img2img_script_container #script_list input");
    const optionTemplate = document.querySelector("#c9-options-template");
    if (optionTemplate && scriptInput.dataset.fixtureOpenBound !== "true") {{
        scriptInput.dataset.fixtureOpenBound = "true";
        scriptInput.addEventListener("keydown", (event) => {{
            if (event.key !== "ArrowDown") return;
            event.preventDefault();
            window.fixtureOptionEvents.openKeydown += 1;
            scriptInput.setAttribute("aria-expanded", "true");
            if (!document.querySelector("#c9-options")) {{
                optionTemplate.before(optionTemplate.content.cloneNode(true));
                bindFixtureControls();
            }}
        }});
    }}
    const listbox = document.querySelector(
        "#img2img_script_container ul[role='listbox'], " +
        "#img2img_script_container div[role='listbox']"
    );
    const scriptOptions = Array.from(
        document.querySelectorAll("#img2img_script_container [role='option']")
    );
    const selectFixtureOption = (option, selectedIndex) => {{
        window.fixtureOptionEvents.selectedIndex = String(selectedIndex);
        scriptInput.value = option.getAttribute("aria-label");
        scriptInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
        scriptInput.dispatchEvent(new Event("change", {{ bubbles: true }}));
        if (scriptInput.value === "Krea2 2-Stage Upscale") {{
            document.querySelector("#script_krea2_2stage_upscale_quick_4k").style.display = "block";
        }}
    }};
    if (listbox?.tagName === "UL" && listbox.dataset.fixtureBound !== "true") {{
        listbox.dataset.fixtureBound = "true";
        listbox.addEventListener("mousedown", (event) => {{
            event.preventDefault();
            window.fixtureOptionEvents.mousedown += 1;
            selectFixtureOption(event.target, event.target.dataset.index);
        }});
        scriptOptions.forEach((option) => option.addEventListener("click", () => {{
            window.fixtureOptionEvents.click += 1;
        }}));
    }} else {{
        scriptOptions.forEach((option, index) => {{
            if (option.dataset.fixtureBound === "true") return;
            option.dataset.fixtureBound = "true";
            option.addEventListener("click", () => {{
                window.fixtureOptionEvents.click += 1;
                selectFixtureOption(option, index + 1);
            }});
        }});
    }}
    const modeButtons = Array.from(
        document.querySelectorAll("#mode_img2img [role='tablist'] > button")
    );
    modeButtons.forEach((button) => {{
        if (button.dataset.fixtureBound === "true") return;
        button.dataset.fixtureBound = "true";
        button.addEventListener("click", () => {{
            modeButtons.forEach((candidate) => {{
                const selected = candidate === button;
                candidate.classList.toggle("selected", selected);
                candidate.setAttribute("aria-selected", String(selected));
            }});
        }});
    }});
}}

function mountLazyControls(panel) {{
    if (!panel) return;
    const lazyKind = panel.dataset.lazyKind;
    if (!lazyKind) {{
        bindFixtureControls();
        return;
    }}
    const template = document.querySelector(`#lazy-${{lazyKind}}-controls`);
    panel.append(template.content.cloneNode(true));
    delete panel.dataset.lazyKind;
    bindFixtureControls();
}}

function ensureFixturePanel(index) {{
    const panelId = panelIds[index];
    let panel = document.getElementById(panelId);
    if (panel) return panel;
    const template = document.querySelector(`#lazy-panel-${{panelId}}`);
    if (!template) return null;
    tabs.append(template.content.cloneNode(true));
    panel = document.getElementById(panelId);
    return panel;
}}

initialButtons.forEach((button, index) => button.addEventListener("click", () => {{
    const selectedPanel = ensureFixturePanel(index);
    mountLazyControls(selectedPanel);
    initialButtons.forEach((candidate, candidateIndex) => {{
        const selected = candidate === button;
        candidate.classList.toggle("selected", selected);
        candidate.setAttribute("aria-selected", String(selected));
        const panel = document.getElementById(panelIds[candidateIndex]);
        if (panel) panel.style.display = selected ? "" : "none";
    }});
    uiTabChangeCallbacks.forEach((callback) => callback());
}}));

window.inputAccordionChecked = (id, checked) => {{
    const target = document.getElementById(id);
    target.visibleCheckbox.checked = checked;
    target.onVisibleCheckboxChange();
}};
bindFixtureControls();

new MutationObserver((records) => {{
    uiUpdateCallbacks.forEach((callback) => callback(records));
}}).observe(document.documentElement, {{ childList: true, subtree: true }});
</script>
<script src="/aikimi_tabs.js"></script>
<script>
setTimeout(() => uiLoadedCallbacks.forEach((callback) => callback()), 0);
(async function report() {{
    const waitFor = async (predicate, attempts = 120) => {{
        for (let attempt = 0; attempt < attempts; attempt += 1) {{
            const value = predicate();
            if (value) return value;
            await new Promise((resolve) => setTimeout(resolve, 20));
        }}
        return null;
    }};
    const aliasesReady = await waitFor(() =>
        document.querySelectorAll("#aikimi-feature-nav > .aikimi-feature-nav__button").length === 4
    );
    const eventCounts = {{ krea2: 0, anima38: 0, cleared: 0 }};
    document.addEventListener("aikimi:feature-tab-change", (event) => {{
        if (event.detail.feature === null) eventCounts.cleared += 1;
        if (event.detail.ready && Object.hasOwn(eventCounts, event.detail.feature)) {{
            eventCounts[event.detail.feature] += 1;
        }}
    }});

    const krea = document.querySelector("#aikimi-tab-krea2");
    const anima = document.querySelector("#aikimi-tab-anima38");
    const txt2img = fixtureNativeButton("tab_txt2img");
    const img2img = fixtureNativeButton("tab_img2img");
    const controlsBeforeClicks = {{
        krea: Boolean(document.querySelector("#img2img_script_container #script_list input")),
        anima: Boolean(document.querySelector("#aikimi-txt2img-anima38"))
    }};
    const mountedPanelsBeforeClicks = Array.from(
        tabs.querySelectorAll(":scope > .tabitem")
    ).map((panel) => panel.id);

    let kreaResult = null;
    let kreaClearedAfterModeChange = null;
    let kreaClearEvents = null;
    let clearAfterNative = null;
    let animaResult = null;
    let animaClearedAfterCollapse = null;
    let animaClearEvents = null;
    let keyboardResult = null;
    if (krea && !krea.hidden && anima && !anima.hidden && txt2img && img2img) {{
        krea.click();
        await waitFor(() => window.AikimiTabs.getActiveFeature() === "krea2");
        kreaResult = {{
            active: window.AikimiTabs.getActiveFeature(),
            selectedScript: scriptInput.value,
            panelMounted: Boolean(document.querySelector("#tab_img2img")),
            normalModeSelected: Boolean(document.querySelector(
                "#mode_img2img [role='tablist'] > button:first-child.selected, #mode_img2img > .tab-nav > button:first-child.selected"
            )),
            panelVisible: document.querySelector("#script_krea2_2stage_upscale_quick_4k").offsetParent !== null,
            hostSelected: img2img.classList.contains("selected"),
            hostSubdued: document.querySelector("#aikimi-feature-nav").dataset.activeFeature === "krea2",
            aliasActive: krea.classList.contains("aikimi-feature-active")
        }};
        const beforeKreaClear = eventCounts.cleared;
        document.querySelectorAll("#mode_img2img [role='tablist'] > button")[1].click();
        await waitFor(() => window.AikimiTabs.getActiveFeature() === null);
        kreaClearedAfterModeChange = window.AikimiTabs.getActiveFeature();
        kreaClearEvents = eventCounts.cleared - beforeKreaClear;

        txt2img.click();
        clearAfterNative = window.AikimiTabs.getActiveFeature();

        anima.click();
        await waitFor(() => window.AikimiTabs.getActiveFeature() === "anima38");
        animaResult = {{
            active: window.AikimiTabs.getActiveFeature(),
            open: accordion.querySelector(".label-wrap").classList.contains("open"),
            visibleChecked: accordion.visibleCheckbox.checked,
            hiddenChecked: document.querySelector("#aikimi-txt2img-anima38-checkbox input").checked,
            hostSelected: txt2img.classList.contains("selected")
        }};
        const beforeAnimaClear = eventCounts.cleared;
        accordion.querySelector(".label-wrap").click();
        await waitFor(() => window.AikimiTabs.getActiveFeature() === null);
        animaClearedAfterCollapse = window.AikimiTabs.getActiveFeature();
        animaClearEvents = eventCounts.cleared - beforeAnimaClear;

        txt2img.click();
        const beforeKeyboard = eventCounts.krea2;
        krea.dispatchEvent(new KeyboardEvent("keydown", {{
            key: "Enter", bubbles: true, cancelable: true
        }}));
        await waitFor(() => window.AikimiTabs.getActiveFeature() === "krea2");
        keyboardResult = {{
            activations: eventCounts.krea2 - beforeKeyboard,
            active: window.AikimiTabs.getActiveFeature()
        }};
    }}

    const studioPanelsWereLazyBeforeVisit =
        !document.querySelector("#tab_sensenova_u15_studio") &&
        !document.querySelector("#tab_minimax_h3_studio");
    let nativeLazyResults = null;
    if ({json.dumps(lazy_panels)}) {{
        const sensenova = document.querySelector("#aikimi-tab-sensenova");
        const minimax = document.querySelector("#aikimi-tab-minimax-h3");
        sensenova.click();
        await waitFor(() => window.AikimiTabs.getActiveContainer()?.id === "tab_sensenova_u15_studio");
        const sensenovaResult = {{
            active: window.AikimiTabs.getActiveFeature(),
            container: window.AikimiTabs.getActiveContainer()?.id || null
        }};
        minimax.click();
        await waitFor(() => window.AikimiTabs.getActiveContainer()?.id === "tab_minimax_h3_studio");
        nativeLazyResults = {{
            sensenova: sensenovaResult,
            minimax: {{
                active: window.AikimiTabs.getActiveFeature(),
                container: window.AikimiTabs.getActiveContainer()?.id || null
            }}
        }};
    }}

    window.AikimiTabs?.refresh();
    window.AikimiTabs?.refresh();
    await new Promise((resolve) => setTimeout(resolve, 50));
    const rowBeforeRepair = document.querySelector("#aikimi-feature-nav");
    rowBeforeRepair?.remove();
    const repairedRow = await waitFor(() => {{
        const row = document.querySelector("#aikimi-feature-nav");
        return row && row !== rowBeforeRepair ? row : null;
    }});
    const featureButtons = Array.from(
        document.querySelectorAll("#aikimi-feature-nav > .aikimi-feature-nav__button")
    );
    const nativeAfter = nativeSnapshot();
    const result = document.querySelector("#result");
    result.dataset.ready = "true";
    result.textContent = JSON.stringify({{
        aliasesReady: Boolean(aliasesReady),
        featureLabels: featureButtons.map((button) => button.textContent.trim()),
        kreaCount: document.querySelectorAll("#aikimi-tab-krea2").length,
        animaCount: document.querySelectorAll("#aikimi-tab-anima38").length,
        kreaHidden: Boolean(document.querySelector("#aikimi-tab-krea2")?.hidden),
        animaHidden: Boolean(document.querySelector("#aikimi-tab-anima38")?.hidden),
        kreaResult,
        kreaClearedAfterModeChange,
        kreaClearEvents,
        optionEvents: window.fixtureOptionEvents,
        clearAfterNative,
        animaResult,
        animaClearedAfterCollapse,
        animaClearEvents,
        keyboardResult,
        domMode: {json.dumps("legacy" if legacy_dom else "gradio6")},
        lazyControls: {json.dumps(lazy_controls)},
        lazyPanels: {json.dumps(lazy_panels)},
        controlsBeforeClicks,
        mountedPanelsBeforeClicks,
        studioPanelsWereLazyBeforeVisit,
        nativeLazyResults,
        tabListClass: tabList.className,
        mutationRepairCount: repairedRow ? document.querySelectorAll("#aikimi-feature-nav").length : 0,
        externalRowBeforeTabs: document.querySelector("#aikimi-feature-nav")?.nextElementSibling === tabs,
        nativeCountBefore: nativeBefore.length,
        nativeCountAfter: nativeAfter.length,
        nativeOrderUnchanged: JSON.stringify(nativeAfter) === JSON.stringify(nativeBefore),
        nativeNodeIdentityUnchanged: currentNativeButtons().every(
            (button, index) => button === initialButtons[index]
        ),
        nativeContainsAikimiButtons: currentNativeButtons()
            .some((button) => button.id?.startsWith("aikimi-tab-")),
        nativeStudioButtonsHidden: {{
            sensenova: getComputedStyle(fixtureNativeButton("tab_sensenova_u15_studio")).display === "none",
            minimax: getComputedStyle(fixtureNativeButton("tab_minimax_h3_studio")).display === "none"
        }},
        ariaTargetsExist: Array.from(
            document.querySelectorAll("#aikimi-feature-nav > .aikimi-feature-nav__button:not([hidden])")
        )
            .every((button) => Boolean(document.getElementById(button.getAttribute("aria-controls")))),
        errors: window.fixtureErrors
    }});
}})();
</script></body></html>"""
        self.send_bytes(document.encode(), "text/html; charset=utf-8")


class AikimiTabsChromiumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chrome or Chromium is not installed")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), AikimiTabsFixtureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def render_fixture(self, query=""):
        url = self.base_url + (f"?{query}" if query else "")
        with tempfile.TemporaryDirectory(prefix="aikimi-tabs-chrome-") as profile:
            completed = subprocess.run(  # noqa: S603 - executable is resolved from trusted local paths.
                [
                    self.chromium,
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--window-size=1280,900",
                    f"--user-data-dir={profile}",
                    "--virtual-time-budget=7000",
                    "--dump-dom",
                    url,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        match = re.search(r'<pre id="result"[^>]*>(.*?)</pre>', completed.stdout, re.DOTALL)
        self.assertIsNotNone(match, completed.stdout[-2000:])
        self.assertIn('data-ready="true"', match.group(0))
        return json.loads(html.unescape(match.group(1)))

    def test_aliases_drive_real_forge_controls_once(self):
        result = self.render_fixture()

        self.assertEqual(result["domMode"], "gradio6")
        self.assertEqual(result["tabListClass"], "tab-container")
        self.assertTrue(result["aliasesReady"], result)
        self.assertEqual(result["kreaCount"], 1)
        self.assertEqual(result["animaCount"], 1)
        self.assertEqual(
            result["featureLabels"],
            ["Krea2", "Anima", "SenseNova", "MiniMax H3"],
        )
        self.assertTrue(result["externalRowBeforeTabs"])
        self.assertEqual(result["nativeCountAfter"], result["nativeCountBefore"])
        self.assertTrue(result["nativeOrderUnchanged"])
        self.assertTrue(result["nativeNodeIdentityUnchanged"])
        self.assertFalse(result["nativeContainsAikimiButtons"])
        self.assertEqual(
            result["nativeStudioButtonsHidden"],
            {"sensenova": True, "minimax": True},
        )
        self.assertTrue(result["ariaTargetsExist"])
        self.assertEqual(result["kreaResult"]["active"], "krea2")
        self.assertEqual(result["kreaResult"]["selectedScript"], "Krea2 2-Stage Upscale")
        self.assertEqual(
            result["optionEvents"],
            {"openKeydown": 1, "mousedown": 1, "click": 0, "selectedIndex": "1"},
        )
        self.assertTrue(result["kreaResult"]["normalModeSelected"])
        self.assertTrue(result["kreaResult"]["panelVisible"])
        self.assertTrue(result["kreaResult"]["hostSelected"])
        self.assertTrue(result["kreaResult"]["hostSubdued"])
        self.assertTrue(result["kreaResult"]["aliasActive"])
        self.assertIsNone(result["kreaClearedAfterModeChange"])
        self.assertEqual(result["kreaClearEvents"], 1)
        self.assertIsNone(result["clearAfterNative"])
        self.assertEqual(result["animaResult"]["active"], "anima38")
        self.assertTrue(result["animaResult"]["open"])
        self.assertTrue(result["animaResult"]["visibleChecked"])
        self.assertTrue(result["animaResult"]["hiddenChecked"])
        self.assertTrue(result["animaResult"]["hostSelected"])
        self.assertIsNone(result["animaClearedAfterCollapse"])
        self.assertEqual(result["animaClearEvents"], 1)
        self.assertEqual(result["keyboardResult"], {"activations": 1, "active": "krea2"})
        self.assertEqual(result["mutationRepairCount"], 1)

    def test_hidden_base_tab_hides_its_external_shortcut(self):
        result = self.render_fixture("hide=img2img")

        self.assertEqual(result["kreaCount"], 1)
        self.assertEqual(result["animaCount"], 1)
        self.assertTrue(result["kreaHidden"])
        self.assertFalse(result["animaHidden"])
        self.assertTrue(result["ariaTargetsExist"])
        self.assertTrue(result["nativeOrderUnchanged"])

    def test_aliases_open_base_tabs_before_lazy_controls_mount(self):
        result = self.render_fixture("lazy=1")

        self.assertTrue(result["lazyControls"])
        self.assertEqual(result["controlsBeforeClicks"], {"krea": False, "anima": False})
        self.assertTrue(result["aliasesReady"], result)
        self.assertEqual(result["kreaCount"], 1)
        self.assertEqual(result["animaCount"], 1)
        self.assertEqual(result["kreaResult"]["selectedScript"], "Krea2 2-Stage Upscale")
        self.assertTrue(result["kreaResult"]["normalModeSelected"])
        self.assertTrue(result["kreaResult"]["panelVisible"])
        self.assertTrue(result["animaResult"]["visibleChecked"])
        self.assertTrue(result["animaResult"]["hiddenChecked"])
        self.assertEqual(result["errors"], [])

    def test_aliases_exist_before_unselected_gradio6_panels_mount(self):
        result = self.render_fixture("lazy_panels=1&lazy=1")

        self.assertTrue(result["lazyPanels"])
        self.assertEqual(result["mountedPanelsBeforeClicks"], ["tab_txt2img"])
        self.assertEqual(result["controlsBeforeClicks"], {"krea": False, "anima": False})
        self.assertTrue(result["aliasesReady"], result)
        self.assertEqual(result["kreaCount"], 1)
        self.assertEqual(result["animaCount"], 1)
        self.assertEqual(
            result["featureLabels"],
            ["Krea2", "Anima", "SenseNova", "MiniMax H3"],
        )
        self.assertEqual(result["nativeCountAfter"], result["nativeCountBefore"])
        self.assertTrue(result["nativeOrderUnchanged"])
        self.assertTrue(result["nativeNodeIdentityUnchanged"])
        self.assertFalse(result["nativeContainsAikimiButtons"])
        self.assertTrue(result["kreaResult"]["panelMounted"])
        self.assertEqual(result["kreaResult"]["selectedScript"], "Krea2 2-Stage Upscale")
        self.assertTrue(result["animaResult"]["visibleChecked"])
        self.assertTrue(result["animaResult"]["hiddenChecked"])
        self.assertTrue(result["studioPanelsWereLazyBeforeVisit"])
        self.assertEqual(
            result["nativeLazyResults"]["sensenova"],
            {"active": "sensenova", "container": "tab_sensenova_u15_studio"},
        )
        self.assertEqual(
            result["nativeLazyResults"]["minimax"],
            {"active": "minimax_h3", "container": "tab_minimax_h3_studio"},
        )
        self.assertEqual(result["errors"], [])

    def test_legacy_direct_tab_nav_remains_supported(self):
        result = self.render_fixture("dom=legacy")

        self.assertEqual(result["domMode"], "legacy")
        self.assertEqual(result["tabListClass"], "tab-nav")
        self.assertEqual(result["kreaCount"], 1)
        self.assertEqual(result["animaCount"], 1)
        self.assertEqual(result["kreaResult"]["active"], "krea2")
        self.assertEqual(result["animaResult"]["active"], "anima38")
        self.assertEqual(
            result["optionEvents"],
            {"openKeydown": 0, "mousedown": 0, "click": 1, "selectedIndex": "1"},
        )
        self.assertEqual(result["mutationRepairCount"], 1)
        self.assertTrue(result["externalRowBeforeTabs"])
        self.assertTrue(result["nativeOrderUnchanged"])
        self.assertEqual(result["errors"], [])


if __name__ == "__main__":
    unittest.main()
