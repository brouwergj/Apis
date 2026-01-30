using UnityEngine;

public class LookAtTarget : MonoBehaviour
{
    [Tooltip("The GameObject this camera should look at")]
    public Transform target;

    [Tooltip("If true, updates in LateUpdate (recommended for cameras)")]
    public bool useLateUpdate = true;

    void Update()
    {
        if (!useLateUpdate)
            LookAt();
    }

    void LateUpdate()
    {
        if (useLateUpdate)
            LookAt();
    }

    void LookAt()
    {
        if (target == null)
            return;

        transform.LookAt(target);
    }
}
