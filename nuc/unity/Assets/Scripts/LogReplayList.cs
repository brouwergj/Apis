using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class LogReplayList : MonoBehaviour
{
    [Header("UI")]
    public RectTransform contentRoot;
    public Button itemButtonPrefab;

    [Header("Replay")]
    public LogReplayPlayer replayPlayer;
    public Button replayButton;

    [Header("Log Sources")]
    public string logsRoot = ""; // optional override
    public string[] subfolders = { "toy_controller", "crazyflie_controller" };

    [Header("Refresh")]
    public bool populateOnStart = true;
    public bool pollForChanges = true;
    public float pollIntervalSeconds = 1.0f;

    private float _nextPollTime = 0f;
    private List<string> _currentFiles = new List<string>();
    private string _selectedPath = "";

    private void Start()
    {
        if (populateOnStart)
            RefreshList();

        if (replayButton != null)
            replayButton.onClick.AddListener(UI_ReplaySelected);
    }

    private void Update()
    {
        if (!pollForChanges)
            return;

        if (Time.realtimeSinceStartup < _nextPollTime)
            return;

        _nextPollTime = Time.realtimeSinceStartup + pollIntervalSeconds;
        RefreshList();
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
            files.AddRange(Directory.GetFiles(dir, "*.csv", SearchOption.TopDirectoryOnly));
        }

        files = files.OrderByDescending(File.GetLastWriteTimeUtc).ToList();

        if (IsSameList(files, _currentFiles))
            return;

        _currentFiles = files;
        RebuildList(files);
    }

    private void RebuildList(List<string> files)
    {
        ClearList();

        if (contentRoot == null || itemButtonPrefab == null)
            return;

        foreach (var path in files)
        {
            var btn = Instantiate(itemButtonPrefab, contentRoot);
            var label = btn.GetComponentInChildren<TextMeshProUGUI>();
            if (label != null)
            {
                label.text = Path.GetFileName(path);
            }

            btn.onClick.AddListener(() => OnItemSelected(path));
        }

        if (files.Count > 0)
        {
            OnItemSelected(files[0]);
        }
    }

    private void OnItemSelected(string path)
    {
        _selectedPath = path;
        if (replayPlayer != null)
            replayPlayer.LoadLog(path);
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
    }

    private void ClearList()
    {
        if (contentRoot == null)
            return;

        for (int i = contentRoot.childCount - 1; i >= 0; i--)
        {
            Destroy(contentRoot.GetChild(i).gameObject);
        }
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
