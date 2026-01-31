using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class LogReplayList : MonoBehaviour
{
    [Header("UI")]
    public RectTransform contentRoot;
    public ScrollRect scrollRect;
    public ToggleGroup toggleGroup;
    public LogReplayListItem itemPrefab;

    [Header("Replay")]
    public LogReplayPlayer replayPlayer;
    public Button replayButton;
    public Button refreshButton;
    public ControllerOrchestrator orchestrator;

    [Header("Log Sources")]
    public string logsRoot = ""; // optional override
    public string[] subfolders = { "toy_controller", "crazyflie_controller" };

    [Header("Refresh")]
    public bool populateOnStart = true;
    private List<string> _currentFiles = new List<string>();
    private readonly List<LogReplayListItem> _items = new List<LogReplayListItem>();
    private string _selectedPath = "";

    private void Start()
    {
        if (populateOnStart)
            RefreshList();

        if (replayButton != null)
            replayButton.onClick.AddListener(UI_ReplaySelected);

        if (refreshButton != null)
            refreshButton.onClick.AddListener(RefreshList);
    }

    public void RefreshList()
    {
        string root = ResolveLogsRoot();
        if (!Directory.Exists(root))
        {
            Debug.LogWarning($"[LogReplayList] Logs root not found: {root}");
            ClearList();
            return;
        }

        var files = new List<string>();
        foreach (var folder in subfolders ?? Array.Empty<string>())
        {
            if (string.IsNullOrWhiteSpace(folder))
                continue;
            string dir = Path.Combine(root, folder);
            if (!Directory.Exists(dir))
                continue;
            var folderFiles = Directory.GetFiles(dir, "*.csv", SearchOption.TopDirectoryOnly);
            Debug.Log($"[LogReplayList] Found {folderFiles.Length} log(s) in {dir}");
            foreach (var file in folderFiles)
                Debug.Log($"[LogReplayList] {Path.GetFileName(file)}");
            files.AddRange(folderFiles);
        }

        files = files.OrderByDescending(File.GetLastWriteTimeUtc).ToList();

        if (IsSameList(files, _currentFiles))
            return;

        Debug.Log($"[LogReplayList] Total logs: {files.Count}");
        _currentFiles = files;
        RebuildList(files);
    }

    private void RebuildList(List<string> files)
    {
        ClearList();

        EnsureContentRoot();
        EnsureToggleGroup();
        if (contentRoot == null || itemPrefab == null)
        {
            Debug.LogWarning($"[LogReplayList] Missing UI refs. contentRoot={(contentRoot != null)}, itemPrefab={(itemPrefab != null)}");
            return;
        }

        foreach (var path in files)
        {
            var item = Instantiate(itemPrefab, contentRoot);
            item.Init(path);
            _items.Add(item);

            if (item.toggle != null)
            {
                item.toggle.group = toggleGroup;
                item.toggle.onValueChanged.AddListener(isOn => OnItemToggleChanged(item, isOn));
            }
        }

        if (files.Count > 0)
        {
            SelectItem(_items[0], true);
        }
    }

    private void OnItemToggleChanged(LogReplayListItem item, bool isOn)
    {
        if (!isOn)
            return;

        SelectItem(item, false);
    }

    private void SelectItem(LogReplayListItem item, bool forceToggleOn)
    {
        if (item == null)
            return;

        if (forceToggleOn && item.toggle != null && !item.toggle.isOn)
            item.toggle.SetIsOnWithoutNotify(true);

        _selectedPath = item.Path;
        if (replayPlayer != null && !string.IsNullOrWhiteSpace(_selectedPath))
            replayPlayer.LoadLog(_selectedPath);
    }

    public void UI_ReplaySelected()
    {
        if (replayPlayer == null)
            return;

        if (string.IsNullOrWhiteSpace(_selectedPath))
        {
            Debug.LogWarning("[LogReplayList] No log selected for replay.");
            return;
        }

        replayPlayer.LoadLog(_selectedPath);
        replayPlayer.Replay();

        if (orchestrator != null)
            orchestrator.SetReplayMode(true);
    }

    private void ClearList()
    {
        EnsureContentRoot();
        if (contentRoot == null)
            return;

        for (int i = contentRoot.childCount - 1; i >= 0; i--)
        {
            Destroy(contentRoot.GetChild(i).gameObject);
        }
        _items.Clear();
    }

    private void EnsureContentRoot()
    {
        if (scrollRect != null)
        {
            if (scrollRect.content != null)
            {
                contentRoot = scrollRect.content;
                return;
            }
        }

        if (contentRoot == null)
            return;

        // If contentRoot points to a ScrollRect or its viewport, prefer the ScrollRect content.
        var sr = contentRoot.GetComponent<ScrollRect>();
        if (sr != null && sr.content != null)
        {
            contentRoot = sr.content;
        }
    }

    private void EnsureToggleGroup()
    {
        if (toggleGroup != null)
            return;

        if (contentRoot != null)
            toggleGroup = contentRoot.GetComponent<ToggleGroup>();

        if (toggleGroup == null && contentRoot != null)
            toggleGroup = contentRoot.gameObject.AddComponent<ToggleGroup>();
    }

    private string ResolveLogsRoot()
    {
        if (!string.IsNullOrWhiteSpace(logsRoot))
            return logsRoot;

        // Assets -> unity -> nuc -> logs
        string path = Path.Combine(Application.dataPath, "..", "..", "logs");
        return Path.GetFullPath(path);
    }

    private static bool IsSameList(List<string> a, List<string> b)
    {
        if (a == null && b == null)
            return true;
        if (a == null || b == null)
            return false;
        if (a.Count != b.Count)
            return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (!string.Equals(a[i], b[i], StringComparison.OrdinalIgnoreCase))
                return false;
        }
        return true;
    }
}
