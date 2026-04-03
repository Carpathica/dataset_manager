"use strict";

const el = {
  imagesDirInput: document.getElementById("imagesDirInput"),
  labelsDirInput: document.getElementById("labelsDirInput"),
  classesFileInput: document.getElementById("classesFileInput"),
  loadSessionBtn: document.getElementById("loadSessionBtn"),
  rescanBtn: document.getElementById("rescanBtn"),
  sessionSummary: document.getElementById("sessionSummary"),
  statImages: document.getElementById("statImages"),
  statLabels: document.getElementById("statLabels"),
  statPairs: document.getElementById("statPairs"),
  deleteAllImagesCheck: document.getElementById("deleteAllImagesCheck"),
  deleteAllLabelsCheck: document.getElementById("deleteAllLabelsCheck"),
  orphanImagesBox: document.getElementById("orphanImagesBox"),
  orphanLabelsBox: document.getElementById("orphanLabelsBox"),
  applyOrphanActionsBtn: document.getElementById("applyOrphanActionsBtn"),
  resolvePairDuplicatesBtn: document.getElementById("resolvePairDuplicatesBtn"),
  pairDuplicatesBox: document.getElementById("pairDuplicatesBox"),
  moveLabelsTargetInput: document.getElementById("moveLabelsTargetInput"),
  moveUnpairedCheck: document.getElementById("moveUnpairedCheck"),
  moveLabelsModeInput: document.getElementById("moveLabelsModeInput"),
  moveLabelsBtn: document.getElementById("moveLabelsBtn"),
  trainRatioInput: document.getElementById("trainRatioInput"),
  valRatioInput: document.getElementById("valRatioInput"),
  testRatioInput: document.getElementById("testRatioInput"),
  splitSeedInput: document.getElementById("splitSeedInput"),
  splitModeInput: document.getElementById("splitModeInput"),
  splitOnlyPairedCheck: document.getElementById("splitOnlyPairedCheck"),
  splitOutputInput: document.getElementById("splitOutputInput"),
  splitYamlCheck: document.getElementById("splitYamlCheck"),
  splitYamlNameInput: document.getElementById("splitYamlNameInput"),
  splitBtn: document.getElementById("splitBtn"),
  mergeSourcesInput: document.getElementById("mergeSourcesInput"),
  addMergeSourceBtn: document.getElementById("addMergeSourceBtn"),
  mergePreviewBtn: document.getElementById("mergePreviewBtn"),
  mergeOutputInput: document.getElementById("mergeOutputInput"),
  mergeModeInput: document.getElementById("mergeModeInput"),
  mergeDeleteSkippedCheck: document.getElementById("mergeDeleteSkippedCheck"),
  mergeYamlCheck: document.getElementById("mergeYamlCheck"),
  mergeRunBtn: document.getElementById("mergeRunBtn"),
  mergeDuplicatesBox: document.getElementById("mergeDuplicatesBox"),
  analyticsOutputInput: document.getElementById("analyticsOutputInput"),
  analyticsBtn: document.getElementById("analyticsBtn"),
  chartsBox: document.getElementById("chartsBox"),
  statusBar: document.getElementById("statusBar"),
  pickerModal: document.getElementById("pickerModal"),
  pickerTitle: document.getElementById("pickerTitle"),
  pickerCloseBtn: document.getElementById("pickerCloseBtn"),
  pickerRootsBtn: document.getElementById("pickerRootsBtn"),
  pickerUpBtn: document.getElementById("pickerUpBtn"),
  pickerSelectCurrentBtn: document.getElementById("pickerSelectCurrentBtn"),
  pickerPath: document.getElementById("pickerPath"),
  pickerList: document.getElementById("pickerList"),
  imagePreviewModal: document.getElementById("imagePreviewModal"),
  imagePreviewTitle: document.getElementById("imagePreviewTitle"),
  imagePreviewPath: document.getElementById("imagePreviewPath"),
  imagePreviewImg: document.getElementById("imagePreviewImg"),
  imagePreviewText: document.getElementById("imagePreviewText"),
  imagePreviewCloseBtn: document.getElementById("imagePreviewCloseBtn"),
};

const state = {
  session: null,
  scan: null,
  mergePreview: null,
  lastPickerPath: null,
  picker: {
    targetInputId: null,
    mode: "dir",
    currentPath: null,
    parentPath: null,
    appendLine: false,
  },
};

function setStatus(message, isError = false) {
  el.statusBar.textContent = message;
  el.statusBar.classList.toggle("error", isError);
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (error) {
      throw new Error(`Invalid server response (${response.status})`);
    }
  }
  if (!response.ok) {
    const detail = data.detail || `${response.status} ${response.statusText}`;
    throw new Error(String(detail));
  }
  return data;
}

function parseLines(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function composeAbsolutePath(base, rel) {
  const basePath = String(base || "").trim();
  const relPath = String(rel || "").trim();
  if (!basePath) {
    return relPath;
  }
  const useBackslash = basePath.includes("\\");
  const sep = useBackslash ? "\\" : "/";
  const normalizedRel = relPath.replace(/[\\/]+/g, sep);
  const needsSep = !(basePath.endsWith("\\") || basePath.endsWith("/"));
  return `${basePath}${needsSep ? sep : ""}${normalizedRel}`;
}

function updateSessionSummary(session, scan) {
  if (!session) {
    el.sessionSummary.textContent = "Dataset is not connected.";
    return;
  }
  const labelsDir = session.labels_dir || "(same folder as images)";
  const classesInfo = (session.classes || []).length;
  const classesFile = session.classes_file || "(auto)";
  el.sessionSummary.innerHTML = [
    `<div><strong>images:</strong> ${session.images_dir || "-"}</div>`,
    `<div><strong>labels:</strong> ${labelsDir}</div>`,
    `<div><strong>classes:</strong> ${classesInfo} (${classesFile})</div>`,
    `<div><strong>pairs:</strong> ${(scan && scan.pair_count) || 0}</div>`,
    `<div><strong>duplicates:</strong> ${(scan && scan.duplicate_group_count) || 0} groups</div>`,
  ].join("");
}

function buildOrphanTable(kind, items) {
  const box = kind === "image" ? el.orphanImagesBox : el.orphanLabelsBox;
  box.innerHTML = "";

  if (!Array.isArray(items) || items.length === 0) {
    const empty = document.createElement("div");
    empty.textContent = "No files.";
    box.appendChild(empty);
    return;
  }

  const table = document.createElement("table");
  table.className = "mini-table";
  const thead = document.createElement("thead");
  const hRow = document.createElement("tr");
  ["Path", "Delete"].forEach((name) => {
    const th = document.createElement("th");
    th.textContent = name;
    hRow.appendChild(th);
  });
  thead.appendChild(hRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  items.forEach((relPath) => {
    const tr = document.createElement("tr");
    tr.dataset.kind = kind;
    tr.dataset.relPath = relPath;

    const pathCell = document.createElement("td");
    const pathBtn = document.createElement("button");
    pathBtn.type = "button";
    pathBtn.className = "dup-path-link";
    pathBtn.textContent = relPath;
    pathBtn.addEventListener("click", () => {
      const root = kind === "image" ? state.scan && state.scan.images_root : state.scan && state.scan.labels_root;
      const absolutePath = composeAbsolutePath(root, relPath);
      if (kind === "image") {
        openImagePreview(absolutePath, relPath);
      } else {
        openTextPreview(absolutePath, relPath);
      }
    });
    pathCell.appendChild(pathBtn);
    tr.appendChild(pathCell);

    const checkboxCell = document.createElement("td");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "orphan-delete-check";
    checkboxCell.appendChild(checkbox);
    tr.appendChild(checkboxCell);

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  box.appendChild(table);
}

function applyDeleteAll(kind, checked) {
  const rows = Array.from(document.querySelectorAll(`tr[data-kind="${kind}"]`));
  rows.forEach((row) => {
    const checkbox = row.querySelector("input.orphan-delete-check");
    if (checkbox) {
      checkbox.checked = checked;
    }
  });
}

function createDupLink(label, targetPath, previewType = "image", titleHint = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "dup-path-link";
  button.textContent = label;
  button.title = titleHint || label;
  button.addEventListener("click", () => {
    if (previewType === "text") {
      openTextPreview(targetPath, label);
      return;
    }
    openImagePreview(targetPath, label);
  });
  return button;
}

function renderPairDuplicates(scan) {
  el.pairDuplicatesBox.innerHTML = "";
  const groups = scan.duplicate_groups || [];
  if (groups.length === 0) {
    el.pairDuplicatesBox.textContent = "No duplicates found.";
    return;
  }

  groups.forEach((group) => {
    const card = document.createElement("div");
    card.className = "dup-card";
    card.dataset.groupId = group.group_id;

    const title = document.createElement("div");
    title.className = "dup-card-title";
    title.textContent = `Group ${group.group_id.slice(0, 10)}... (${group.count} files)`;
    card.appendChild(title);

    (group.entries || []).forEach((entry, index) => {
      const item = document.createElement("div");
      item.className = "dup-item";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = `pair-dup-${group.group_id}`;
      radio.value = String(index);
      radio.checked = index === 0;
      item.appendChild(radio);

      const imageLine = document.createElement("div");
      imageLine.appendChild(
        createDupLink(entry.image_rel || "(image)", entry.image_path, "image", entry.image_path || "")
      );
      item.appendChild(imageLine);

      const labelLine = document.createElement("div");
      labelLine.style.marginTop = "4px";
      if (entry.label_rel && entry.label_path) {
        labelLine.appendChild(createDupLink(entry.label_rel, entry.label_path, "text", entry.label_path));
      } else {
        labelLine.textContent = "(no label)";
      }
      item.appendChild(labelLine);

      card.appendChild(item);
    });
    el.pairDuplicatesBox.appendChild(card);
  });
}

function renderMergeDuplicates(preview) {
  el.mergeDuplicatesBox.innerHTML = "";
  const groups = preview.duplicate_groups || [];
  if (groups.length === 0) {
    el.mergeDuplicatesBox.textContent = "No hash duplicates found.";
    return;
  }

  groups.forEach((group) => {
    const card = document.createElement("div");
    card.className = "dup-card";
    card.dataset.groupId = group.group_id;
    const title = document.createElement("div");
    title.className = "dup-card-title";
    title.textContent = `Group ${group.group_id.slice(0, 10)}... (${group.count} files)`;
    card.appendChild(title);

    (group.entries || []).forEach((entry, index) => {
      const item = document.createElement("div");
      item.className = "dup-item";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = `merge-dup-${group.group_id}`;
      radio.value = String(index);
      radio.checked = index === 0;
      item.appendChild(radio);

      const imageLine = document.createElement("div");
      imageLine.appendChild(
        createDupLink(
          `${entry.dataset_dir} | ${entry.split}/${entry.inner_key}`,
          entry.image_path,
          "image",
          entry.image_path || ""
        )
      );
      item.appendChild(imageLine);

      if (entry.label_path) {
        const labelLine = document.createElement("div");
        labelLine.style.marginTop = "4px";
        labelLine.appendChild(createDupLink(entry.label_path, entry.label_path, "text", entry.label_path));
        item.appendChild(labelLine);
      }

      card.appendChild(item);
    });
    el.mergeDuplicatesBox.appendChild(card);
  });
}

function renderScan(scan) {
  state.scan = scan;
  el.statImages.textContent = `images: ${scan.image_count}`;
  el.statLabels.textContent = `labels: ${scan.label_count}`;
  el.statPairs.textContent = `pairs: ${scan.pair_count}`;
  buildOrphanTable("image", scan.images_without_labels || []);
  buildOrphanTable("label", scan.labels_without_images || []);
  renderPairDuplicates(scan);
  el.deleteAllImagesCheck.checked = false;
  el.deleteAllLabelsCheck.checked = false;
}

function setImagePreviewVisible(visible) {
  el.imagePreviewModal.classList.toggle("hidden", !visible);
}

function closeImagePreview() {
  el.imagePreviewText.classList.add("hidden");
  el.imagePreviewText.textContent = "";
  el.imagePreviewImg.classList.remove("hidden");
  el.imagePreviewImg.removeAttribute("src");
  el.imagePreviewPath.textContent = "";
  setImagePreviewVisible(false);
}

function openImagePreview(imagePath, title) {
  if (!imagePath) {
    setStatus("Image path is missing for this record.", true);
    return;
  }
  el.imagePreviewTitle.textContent = title || "Image Preview";
  el.imagePreviewPath.textContent = imagePath;
  el.imagePreviewText.classList.add("hidden");
  el.imagePreviewText.textContent = "";
  el.imagePreviewImg.classList.remove("hidden");
  el.imagePreviewImg.src = `/api/image/by-path?path=${encodeURIComponent(imagePath)}`;
  setImagePreviewVisible(true);
}

async function openTextPreview(textPath, title) {
  if (!textPath) {
    setStatus("Text path is missing for this record.", true);
    return;
  }
  el.imagePreviewTitle.textContent = title || "Text Preview";
  el.imagePreviewPath.textContent = textPath;
  el.imagePreviewImg.classList.add("hidden");
  el.imagePreviewImg.removeAttribute("src");
  el.imagePreviewText.classList.remove("hidden");
  el.imagePreviewText.textContent = "Loading...";
  setImagePreviewVisible(true);

  try {
    const data = await apiJson(`/api/text/by-path?path=${encodeURIComponent(textPath)}`);
    el.imagePreviewPath.textContent = data.path || textPath;
    el.imagePreviewText.textContent = data.content || "";
  } catch (error) {
    el.imagePreviewText.textContent = `Cannot load text file.\n${error.message}`;
    setStatus(`Text preview failed: ${error.message}`, true);
  }
}

async function loadSession() {
  const payload = {
    images_dir: el.imagesDirInput.value.trim(),
    labels_dir: el.labelsDirInput.value.trim() || null,
    classes_file: el.classesFileInput.value.trim() || null,
  };
  if (!payload.images_dir) {
    setStatus("Set images folder first.", true);
    return;
  }
  setStatus("Connecting dataset...");
  try {
    const data = await apiJson("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.session = data.session;
    renderScan(data.scan);
    updateSessionSummary(data.session, data.scan);
    el.imagesDirInput.value = data.session.images_dir || payload.images_dir;
    el.labelsDirInput.value = data.session.labels_dir || "";
    el.classesFileInput.value = data.session.classes_file || "";
    setStatus(`Dataset connected. Pairs: ${data.scan.pair_count}`);
  } catch (error) {
    setStatus(`Connection failed: ${error.message}`, true);
  }
}

async function rescanDataset() {
  setStatus("Refreshing analysis...");
  try {
    const data = await apiJson("/api/scan", { method: "POST" });
    state.session = data.session;
    renderScan(data.scan);
    updateSessionSummary(data.session, data.scan);
    setStatus(
      `Done. Orphan images: ${(data.scan.images_without_labels || []).length}, orphan labels: ${(data.scan.labels_without_images || []).length}`
    );
  } catch (error) {
    setStatus(`Scan failed: ${error.message}`, true);
  }
}

function collectOrphanActions() {
  const rows = Array.from(document.querySelectorAll("tr[data-kind][data-rel-path]"));
  return rows.map((row) => {
    const checkbox = row.querySelector("input.orphan-delete-check");
    return {
      kind: row.dataset.kind,
      rel_path: row.dataset.relPath,
      action: checkbox && checkbox.checked ? "delete" : "keep",
    };
  });
}

async function applyOrphanActions() {
  const items = collectOrphanActions();
  if (items.length === 0) {
    setStatus("Orphan lists are empty.");
    return;
  }
  setStatus("Deleting selected orphan files...");
  try {
    const data = await apiJson("/api/orphans/actions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
    renderScan(data.scan);
    updateSessionSummary(state.session, data.scan);
    setStatus(
      `Done. Deleted: ${data.result.deleted_count}, kept: ${data.result.kept_count}, errors: ${(data.result.errors || []).length}`
    );
  } catch (error) {
    setStatus(`Apply failed: ${error.message}`, true);
  }
}

function collectPairDuplicateResolutions() {
  const result = [];
  const cards = Array.from(el.pairDuplicatesBox.querySelectorAll(".dup-card[data-group-id]"));
  cards.forEach((card) => {
    const groupId = card.dataset.groupId;
    const checked = card.querySelector("input[type='radio']:checked");
    if (!groupId || !checked) {
      return;
    }
    result.push({
      group_id: groupId,
      keep_index: Number(checked.value) || 0,
    });
  });
  return result;
}

async function resolvePairDuplicates() {
  const resolutions = collectPairDuplicateResolutions();
  if (resolutions.length === 0) {
    setStatus("No duplicate groups to process.");
    return;
  }
  setStatus("Resolving duplicates...");
  try {
    const data = await apiJson("/api/duplicates/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resolutions }),
    });
    renderScan(data.scan);
    updateSessionSummary(state.session, data.scan);
    setStatus(
      `Duplicates resolved. Groups: ${data.result.group_count}, deleted files: ${data.result.deleted_files_count}, errors: ${(data.result.delete_errors || []).length}`
    );
  } catch (error) {
    setStatus(`Resolve failed: ${error.message}`, true);
  }
}

async function moveLabels() {
  setStatus("Running move/copy labels...");
  try {
    const data = await apiJson("/api/labels/separate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_labels_dir: el.moveLabelsTargetInput.value.trim() || null,
        move_unpaired: Boolean(el.moveUnpairedCheck.checked),
        mode: el.moveLabelsModeInput.value,
      }),
    });
    state.session = data.session;
    renderScan(data.scan);
    updateSessionSummary(data.session, data.scan);
    el.labelsDirInput.value = data.session.labels_dir || "";
    el.moveLabelsTargetInput.value = data.result.target_labels_dir || "";
    setStatus(`${data.result.mode}: processed ${data.result.moved_count}, skipped ${data.result.skipped_count}`);
  } catch (error) {
    setStatus(`Move/copy failed: ${error.message}`, true);
  }
}

async function runSplit() {
  const outputDir = el.splitOutputInput.value.trim();
  if (!outputDir) {
    setStatus("Set split output folder.", true);
    return;
  }
  setStatus("Creating split...");
  try {
    const data = await apiJson("/api/split", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        output_dir: outputDir,
        train_ratio: Number(el.trainRatioInput.value),
        val_ratio: Number(el.valRatioInput.value),
        test_ratio: Number(el.testRatioInput.value),
        seed: Number(el.splitSeedInput.value) || 42,
        mode: el.splitModeInput.value,
        include_only_paired: Boolean(el.splitOnlyPairedCheck.checked),
        generate_yaml: Boolean(el.splitYamlCheck.checked),
        yaml_name: el.splitYamlNameInput.value.trim() || "data.yaml",
      }),
    });
    const result = data.result;
    const counts = result.split_counts || {};
    setStatus(
      `Split ready. train=${counts.train || 0}, val=${counts.val || 0}, test=${counts.test || 0}${
        result.yaml_path ? `, yaml=${result.yaml_path}` : ""
      }`
    );
  } catch (error) {
    setStatus(`Split failed: ${error.message}`, true);
  }
}

function parseMergeSources() {
  return parseLines(el.mergeSourcesInput.value);
}

async function previewMerge() {
  const datasetDirs = parseMergeSources();
  if (datasetDirs.length === 0) {
    setStatus("Add source dataset folders first.", true);
    return;
  }
  setStatus("Finding merge duplicates...");
  try {
    const data = await apiJson("/api/merge/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_dirs: datasetDirs }),
    });
    state.mergePreview = data;
    renderMergeDuplicates(data);
    setStatus(`Preview ready. Items: ${data.total_items}, duplicate groups: ${data.duplicate_group_count}`);
  } catch (error) {
    setStatus(`Merge preview failed: ${error.message}`, true);
  }
}

function collectMergeResolutions() {
  const result = [];
  const cards = Array.from(el.mergeDuplicatesBox.querySelectorAll(".dup-card[data-group-id]"));
  cards.forEach((card) => {
    const groupId = card.dataset.groupId;
    const checked = card.querySelector("input[type='radio']:checked");
    if (!groupId || !checked) {
      return;
    }
    result.push({
      group_id: groupId,
      keep_index: Number(checked.value) || 0,
    });
  });
  return result;
}

async function runMerge() {
  const datasetDirs = parseMergeSources();
  if (datasetDirs.length === 0) {
    setStatus("Add source dataset folders before merge.", true);
    return;
  }
  const outputDir = el.mergeOutputInput.value.trim();
  if (!outputDir) {
    setStatus("Set merged output folder.", true);
    return;
  }
  setStatus("Merging datasets...");
  try {
    const classes = state.session && Array.isArray(state.session.classes) ? state.session.classes : [];
    const mergeYamlEnabled = el.mergeYamlCheck ? Boolean(el.mergeYamlCheck.checked) : false;
    const data = await apiJson("/api/merge/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_dirs: datasetDirs,
        output_dir: outputDir,
        mode: el.mergeModeInput.value,
        resolutions: collectMergeResolutions(),
        delete_skipped_duplicates: Boolean(el.mergeDeleteSkippedCheck.checked),
        generate_yaml: mergeYamlEnabled,
        yaml_name: "data.yaml",
        classes,
      }),
    });
    const result = data.result;
    setStatus(
      `Merge complete. Added: ${result.merged_count}, skipped duplicates: ${result.duplicate_skipped_count}, renamed: ${result.renamed_due_collision}${
        result.yaml_path ? `, yaml=${result.yaml_path}` : ""
      }`
    );
  } catch (error) {
    setStatus(`Merge failed: ${error.message}`, true);
  }
}

function selectedCharts() {
  return Array.from(document.querySelectorAll(".chart-check:checked")).map((item) => item.value);
}

function renderCharts(data) {
  el.chartsBox.innerHTML = "";
  (data.charts || []).forEach((chart) => {
    const card = document.createElement("div");
    card.className = "chart-card";
    const title = document.createElement("h4");
    title.textContent = chart.title;
    const image = document.createElement("img");
    image.src = `data:image/png;base64,${chart.image_base64}`;
    image.alt = chart.title;
    card.appendChild(title);
    card.appendChild(image);
    el.chartsBox.appendChild(card);
  });
}

async function runAnalytics() {
  const charts = selectedCharts();
  if (charts.length === 0) {
    setStatus("Select at least one chart.", true);
    return;
  }
  setStatus("Generating charts...");
  try {
    const data = await apiJson("/api/analytics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        charts,
        output_dir: el.analyticsOutputInput.value.trim() || null,
      }),
    });
    renderCharts(data);
    const summary = data.summary || {};
    const saveInfo = data.output_dir ? `, saved to: ${data.output_dir}` : ", display-only (not saved)";
    setStatus(`Charts ready (${data.chart_count})${saveInfo}. Objects: ${summary.objects_total || 0}`);
  } catch (error) {
    setStatus(`Analytics failed: ${error.message}`, true);
  }
}

function setPickerVisible(visible) {
  el.pickerModal.classList.toggle("hidden", !visible);
}

function closePicker() {
  state.picker = {
    targetInputId: null,
    mode: "dir",
    currentPath: null,
    parentPath: null,
    appendLine: false,
  };
  setPickerVisible(false);
}

function applyPickedPath(path) {
  if (!path) {
    return;
  }
  if (state.picker.appendLine) {
    const lines = parseMergeSources();
    if (!lines.includes(path)) {
      lines.push(path);
      el.mergeSourcesInput.value = lines.join("\n");
    }
    closePicker();
    return;
  }
  const input = document.getElementById(state.picker.targetInputId || "");
  if (input) {
    input.value = path;
  }
  closePicker();
}

function renderPickerList(data) {
  el.pickerList.innerHTML = "";
  el.pickerPath.textContent = data.current_path || "Roots";
  state.picker.currentPath = data.current_path && data.current_path !== "Roots" ? data.current_path : null;
  state.picker.parentPath = data.parent_path || null;
  state.lastPickerPath = state.picker.currentPath || state.lastPickerPath;
  el.pickerUpBtn.disabled = !data.parent_path;
  el.pickerSelectCurrentBtn.style.display = state.picker.mode === "dir" ? "inline-block" : "none";

  (data.directories || []).forEach((entry) => {
    const li = document.createElement("li");
    li.className = "picker-item";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.name;
    button.addEventListener("click", () => loadPickerPath(entry.path, state.picker.mode));
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "DIR";
    li.appendChild(button);
    li.appendChild(tag);
    el.pickerList.appendChild(li);
  });

  (data.files || []).forEach((entry) => {
    const li = document.createElement("li");
    li.className = "picker-item";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.name;
    button.addEventListener("click", () => applyPickedPath(entry.path));
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "FILE";
    li.appendChild(button);
    li.appendChild(tag);
    el.pickerList.appendChild(li);
  });

  if ((data.directories || []).length === 0 && (data.files || []).length === 0) {
    const li = document.createElement("li");
    li.className = "picker-item";
    li.textContent = "No entries";
    el.pickerList.appendChild(li);
  }
}

async function showPickerRoots() {
  try {
    const data = await apiJson("/api/fs/roots");
    renderPickerList({
      current_path: "Roots",
      parent_path: null,
      directories: (data.roots || []).map((path) => ({ name: path, path })),
      files: [],
    });
  } catch (error) {
    setStatus(`Cannot load root folders: ${error.message}`, true);
  }
}

async function loadPickerPath(path, mode) {
  try {
    const data = await apiJson(`/api/fs/list?path=${encodeURIComponent(path)}&mode=${encodeURIComponent(mode)}`);
    renderPickerList(data);
  } catch (error) {
    setStatus(`Path browse error: ${error.message}`, true);
  }
}

function guessPickerStartPath(targetInputId, mode, appendLine) {
  if (appendLine) {
    return state.lastPickerPath || el.imagesDirInput.value.trim() || null;
  }
  const input = document.getElementById(targetInputId);
  if (input) {
    const raw = String(input.value || "").trim();
    if (raw) {
      if (mode === "yaml" && /\.(yaml|yml|txt)$/i.test(raw)) {
        const parts = raw.split(/[\\/]/);
        if (parts.length > 1) {
          parts.pop();
          return parts.join("\\");
        }
      }
      return raw;
    }
  }
  return state.lastPickerPath || el.imagesDirInput.value.trim() || el.labelsDirInput.value.trim() || null;
}

function openPicker(targetInputId, mode, appendLine = false) {
  state.picker = {
    targetInputId,
    mode,
    currentPath: null,
    parentPath: null,
    appendLine,
  };
  el.pickerTitle.textContent = appendLine ? "Select Source Folder" : "Select Path";
  setPickerVisible(true);
  const startPath = guessPickerStartPath(targetInputId, mode, appendLine);
  if (startPath) {
    loadPickerPath(startPath, mode);
  } else {
    showPickerRoots();
  }
}

document.querySelectorAll(".picker-btn").forEach((button) => {
  button.addEventListener("click", () => {
    openPicker(button.dataset.pickerTarget, button.dataset.pickerMode || "dir", false);
  });
});

el.addMergeSourceBtn.addEventListener("click", () => openPicker("mergeSourcesInput", "dir", true));
el.loadSessionBtn.addEventListener("click", loadSession);
el.rescanBtn.addEventListener("click", rescanDataset);
el.applyOrphanActionsBtn.addEventListener("click", applyOrphanActions);
el.deleteAllImagesCheck.addEventListener("change", () => applyDeleteAll("image", el.deleteAllImagesCheck.checked));
el.deleteAllLabelsCheck.addEventListener("change", () => applyDeleteAll("label", el.deleteAllLabelsCheck.checked));
el.resolvePairDuplicatesBtn.addEventListener("click", resolvePairDuplicates);
el.moveLabelsBtn.addEventListener("click", moveLabels);
el.splitBtn.addEventListener("click", runSplit);
el.mergePreviewBtn.addEventListener("click", previewMerge);
el.mergeRunBtn.addEventListener("click", runMerge);
el.analyticsBtn.addEventListener("click", runAnalytics);
el.pickerCloseBtn.addEventListener("click", closePicker);
el.pickerRootsBtn.addEventListener("click", showPickerRoots);
el.pickerUpBtn.addEventListener("click", () => {
  if (state.picker.parentPath) {
    loadPickerPath(state.picker.parentPath, state.picker.mode);
  }
});
el.pickerSelectCurrentBtn.addEventListener("click", () => {
  if (state.picker.currentPath) {
    applyPickedPath(state.picker.currentPath);
  }
});
el.pickerModal.addEventListener("click", (event) => {
  if (event.target === el.pickerModal) {
    closePicker();
  }
});
el.imagePreviewCloseBtn.addEventListener("click", closeImagePreview);
el.imagePreviewModal.addEventListener("click", (event) => {
  if (event.target === el.imagePreviewModal) {
    closeImagePreview();
  }
});
el.imagePreviewImg.addEventListener("error", () => {
  if (!el.imagePreviewImg.classList.contains("hidden")) {
    setStatus("Cannot load image preview for selected item.", true);
  }
});

setStatus("Connect dataset and run analysis.");
