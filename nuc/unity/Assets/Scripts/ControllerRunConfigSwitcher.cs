using System.IO;
using System.Linq;
using UnityEngine;
using TMPro;

public class ControllerRunConfigSwitcher : MonoBehaviour
{
    [Header("Controller Dropdown")]
    public TMP_Dropdown controllerDropdown;

    [Header("RunConfig Dependencies")]
    public RunConfigDropdownPopulator runConfigPopulator;
    public ControllerOrchestrator orchestrator;

    [Header("Options")]
    public bool includePlaceholder = true;
    public string placeholderLabel = "Select Controller";
    public string emptyLabel = "No controllers found";
    public bool includeRootOption = false;
    public string rootLabel = "root";

    [Tooltip("Optional hardcoded controller folder names under StreamingAssets.")]
    public string[] controllerFolders;
    [Tooltip("Folder names to exclude from the controller dropdown.")]
    public string[] excludedFolders = { "root", "toycontroller", "crazyflie_pid", "option_a", "optionA" };

    private void Start()
    {
        PopulateControllers();
    }

    public void PopulateControllers()
    {
        if (controllerDropdown == null)
        {
            Debug.LogWarning("[ControllerRunConfigSwitcher] No controller dropdown assigned.");
            return;
        }

        var options = ResolveControllerOptions();
        if (includePlaceholder)
            options.Insert(0, placeholderLabel);

        controllerDropdown.ClearOptions();
        controllerDropdown.AddOptions(options);
        controllerDropdown.onValueChanged.RemoveListener(OnControllerChanged);
        controllerDropdown.onValueChanged.AddListener(OnControllerChanged);

        if (controllerDropdown.options.Count == 0)
        {
            Debug.LogWarning("[ControllerRunConfigSwitcher] No controller options found.");
            controllerDropdown.ClearOptions();
            controllerDropdown.AddOptions(new System.Collections.Generic.List<string> { emptyLabel });
            controllerDropdown.value = 0;
            controllerDropdown.RefreshShownValue();
            return;
        }

        if (includePlaceholder && controllerDropdown.options.Count > 1)
        {
            // Remove placeholder once we have real options, then select first option.
            controllerDropdown.options.RemoveAt(0);
            controllerDropdown.value = 0;
            controllerDropdown.RefreshShownValue();
            OnControllerChanged(controllerDropdown.value);
        }
        else
        {
            controllerDropdown.value = 0;
            controllerDropdown.RefreshShownValue();
            OnControllerChanged(controllerDropdown.value);
        }
    }

    private System.Collections.Generic.List<string> ResolveControllerOptions()
    {
        var options = new System.Collections.Generic.List<string>();

        if (controllerFolders != null && controllerFolders.Length > 0)
        {
            foreach (var folder in controllerFolders)
            {
                if (!string.IsNullOrWhiteSpace(folder))
                {
                    string name = folder.Trim();
                    if (!IsExcluded(name))
                        options.Add(name);
                }
            }
        }
        else
        {
            string baseDir = Application.streamingAssetsPath;
            if (Directory.Exists(baseDir))
            {
                var dirs = Directory.GetDirectories(baseDir)
                    .Where(d => Directory.GetFiles(d, "*.json", SearchOption.TopDirectoryOnly).Length > 0)
                    .Select(Path.GetFileName)
                    .Where(name => !IsExcluded(name))
                    .OrderBy(d => d);
                options.AddRange(dirs);
            }
        }

        if (includeRootOption)
            options.Insert(0, rootLabel);

        return options;
    }

    private bool IsExcluded(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return true;
        if (excludedFolders == null || excludedFolders.Length == 0)
            return false;
        return excludedFolders.Any(excluded => string.Equals(excluded, name, System.StringComparison.OrdinalIgnoreCase));
    }

    private void OnControllerChanged(int index)
    {
        if (controllerDropdown == null || runConfigPopulator == null)
            return;

        if (controllerDropdown.options == null || controllerDropdown.options.Count == 0)
        {
            Debug.LogWarning("[ControllerRunConfigSwitcher] Controller dropdown has no options.");
            return;
        }

        string label = controllerDropdown.options[index].text;
        if (includePlaceholder && label == placeholderLabel)
            return;

        if (orchestrator != null)
        {
            orchestrator.SetSelectedController(label);
            orchestrator.ResetRunConfigSelection(updateStatus: false);
        }

        string subfolder = label == rootLabel ? "" : label;
        runConfigPopulator.SetSubfolder(subfolder, repopulate: true);
    }
}
