using System.IO;
using UnityEngine;

public class RunConfigCatalog : MonoBehaviour
{
    [Header("Options")]
    public bool printOnStart = true;

    private void Start()
    {
        if (!printOnStart)
            return;

        string dir = ResolveRunConfigsRoot();
        if (!Directory.Exists(dir))
        {
            Debug.LogWarning($"[RunConfigCatalog] RunConfigs folder not found: {dir}");
            return;
        }

        string[] files = Directory.GetFiles(dir, "*.json", SearchOption.TopDirectoryOnly);
        Debug.Log($"[RunConfigCatalog] Found {files.Length} RunConfig file(s) in {dir}");
        foreach (var path in files)
        {
            Debug.Log($"[RunConfigCatalog] {Path.GetFileName(path)}");
        }
    }

    private string ResolveRunConfigsRoot()
    {
        // Assets -> unity -> nuc -> runconfigs
        string path = Path.Combine(Application.dataPath, "..", "..", "runconfigs");
        return Path.GetFullPath(path);
    }
}
