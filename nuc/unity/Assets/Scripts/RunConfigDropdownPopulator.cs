using System.IO;
using System.Linq;
using UnityEngine;
using TMPro;

public class RunConfigDropdownPopulator : MonoBehaviour
{
    [Header("Dropdown")]
    public TMP_Dropdown dropdown;

    [Header("Search")]
    [Tooltip("Optional subfolder under StreamingAssets (e.g., \"toy_controller\"). Leave empty for root.")]
    public string subfolder = "";

    [Header("Options")]
    public bool populateOnStart = true;
    public bool printOnStart = true;
    public bool includePathInLog = false;
    public bool includeSubfolderInLabel = true;
    public string emptyLabel = "No RunConfigs found";
    public bool autoSelectFirst = true;

    [Tooltip("Optional: include a placeholder option at the top.")]
    public bool includePlaceholder = false;
    public string placeholderLabel = "Select RunConfig";

    public string[] RunConfigPaths { get; private set; } = new string[0];
    public string[] RunConfigRelativePaths { get; private set; } = new string[0];

    private void Start()
    {
        if (populateOnStart)
            Populate();
    }

    public void Populate()
    {
        string baseDir = Application.streamingAssetsPath;
        string dir = string.IsNullOrWhiteSpace(subfolder)
            ? baseDir
            : Path.Combine(baseDir, subfolder);
        if (!Directory.Exists(dir))
        {
            Debug.LogWarning($"[RunConfigDropdownPopulator] StreamingAssets folder not found: {dir}");
            RunConfigPaths = new string[0];
            RunConfigRelativePaths = new string[0];
            if (dropdown != null)
            {
                dropdown.ClearOptions();
                if (includePlaceholder)
                    dropdown.AddOptions(new System.Collections.Generic.List<string> { placeholderLabel });
            }
            return;
        }

        var files = Directory.GetFiles(dir, "*.json", SearchOption.TopDirectoryOnly)
            .OrderBy(p => p)
            .ToArray();

        RunConfigPaths = files;
        RunConfigRelativePaths = files.Select(p => MakeRelativePath(baseDir, p)).ToArray();

        if (printOnStart)
        {
            Debug.Log($"[RunConfigDropdownPopulator] Found {files.Length} RunConfig file(s) in {dir}");
            foreach (var path in files)
            {
                string label = includePathInLog ? path : Path.GetFileName(path);
                Debug.Log($"[RunConfigDropdownPopulator] {label}");
            }
        }

        var options = files.Select(p => includeSubfolderInLabel ? MakeRelativePath(baseDir, p) : Path.GetFileName(p)).ToList();
        if (includePlaceholder)
            options.Insert(0, placeholderLabel);

        if (dropdown != null)
        {
            dropdown.ClearOptions();
            if (options.Count > 0)
            {
                dropdown.AddOptions(options);
                if (autoSelectFirst)
                {
                    int index = includePlaceholder ? 1 : 0;
                    if (index >= 0 && index < dropdown.options.Count)
                    {
                        dropdown.value = index;
                        dropdown.RefreshShownValue();
                        dropdown.onValueChanged.Invoke(dropdown.value);
                    }
                }
            }
            else
            {
                Debug.LogWarning("[RunConfigDropdownPopulator] No RunConfig files found.");
                dropdown.AddOptions(new System.Collections.Generic.List<string> { emptyLabel });
            }
        }
        else
        {
            Debug.LogWarning("[RunConfigDropdownPopulator] No TMP_Dropdown assigned.");
        }
    }

    public void SetSubfolder(string newSubfolder, bool repopulate = true)
    {
        subfolder = newSubfolder ?? "";
        if (repopulate)
            Populate();
    }

    public string GetRelativePathForIndex(int index)
    {
        if (RunConfigRelativePaths == null || RunConfigRelativePaths.Length == 0)
            return null;

        int offset = includePlaceholder ? 1 : 0;
        int relIndex = index - offset;
        if (relIndex < 0 || relIndex >= RunConfigRelativePaths.Length)
            return null;
        return RunConfigRelativePaths[relIndex];
    }

    private static string MakeRelativePath(string baseDir, string fullPath)
    {
        var baseUri = new System.Uri(baseDir + Path.DirectorySeparatorChar);
        var fileUri = new System.Uri(fullPath);
        return System.Uri.UnescapeDataString(baseUri.MakeRelativeUri(fileUri).ToString())
            .Replace('/', Path.DirectorySeparatorChar);
    }
}
