using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class LogReplayListItem : MonoBehaviour
{
    public Toggle toggle;
    public TextMeshProUGUI label;

    public string Path { get; private set; } = "";

    public void Init(string path)
    {
        Path = path ?? "";
        if (label != null)
            label.text = System.IO.Path.GetFileName(Path);
    }
}
