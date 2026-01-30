using UnityEngine;

public class PropellerSpinFromTelemetry : MonoBehaviour
{
    [Header("References")]
    public UdpTelemetryReceiver receiver;

    [Tooltip("Assign the 4 propeller transforms (children of the drone GO).")]
    public Transform[] propellers;

    [Header("Spin Mapping")]
    [Tooltip("Max visual RPM when thrust_norm = 1.0")]
    public float maxRpm = 6000f;

    [Tooltip("Minimum RPM at thrust_norm = 0 (so it doesn't look dead)")]
    public float minRpm = 300f;

    [Tooltip("If true, alternate prop direction to mimic quad torque balancing.")]
    public bool alternateDirections = true;

    [Header("Smoothing")]
    public bool smooth = true;
    public float rpmResponse = 25f;

    private float _rpmCurrent;

    private void Update()
    {
        if (receiver == null || propellers == null || propellers.Length == 0)
            return;

        if (!receiver.TryGetLatest(out var pkt))
            return;

        // Prefer thrust_norm; if absent, fall back to thrust (will just be 0)
        float thrustNorm = pkt.thrust_norm;

        // Nonlinear mapping so hover already spins fast
        float x = Mathf.Clamp01(thrustNorm);
        x = Mathf.Pow(x, 0.5f); // boosts low values

        float rpmTarget = Mathf.Lerp(minRpm, maxRpm, x);

        if (smooth)
            _rpmCurrent = Mathf.Lerp(_rpmCurrent, rpmTarget, 1f - Mathf.Exp(-rpmResponse * Time.deltaTime));
        else
            _rpmCurrent = rpmTarget;

        // RPM -> degrees per second: rpm * 360 / 60
        float degPerSec = _rpmCurrent * 6f;
        float deltaDeg = degPerSec * Time.deltaTime;

        for (int i = 0; i < propellers.Length; i++)
        {
            if (propellers[i] == null) continue;

            float dir = 1f;
            if (alternateDirections)
                dir = (i % 2 == 0) ? 1f : -1f;

            // Assumes propellers spin around their local up axis.
            // If your model uses a different axis, change Vector3.up to Vector3.forward/right.
            propellers[i].Rotate(Vector3.up, dir * deltaDeg, Space.Self);
        }
    }
}
