using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public class LogReplayPlayer : MonoBehaviour
{
    [Header("References")]
    public Transform target;

    [Header("Playback")]
    public bool playOnLoad = true;
    public bool loop = false;
    public float timeScale = 1f;

    public string CurrentLogPath { get; private set; } = "";
    public bool IsPlaying { get; private set; } = false;

    private readonly List<Frame> _frames = new List<Frame>();
    private int _frameIndex = 0;
    private float _playbackTime = 0f;
    private float _startTime = 0f;

    private struct Frame
    {
        public float t;
        public Vector3 pos;
        public Quaternion rot;
    }

    private void Reset()
    {
        target = transform;
    }

    public bool LoadLog(string path)
    {
        if (!File.Exists(path))
        {
            Debug.LogWarning($"[LogReplayPlayer] Log not found: {path}");
            return false;
        }

        _frames.Clear();
        _frameIndex = 0;
        _playbackTime = 0f;
        _startTime = 0f;

        try
        {
            using var reader = new StreamReader(path);
            string header = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(header))
                throw new InvalidDataException("Empty log file");

            var cols = header.Split(',');
            int idxT = Array.IndexOf(cols, "t");
            int idxPx = Array.IndexOf(cols, "px");
            int idxPy = Array.IndexOf(cols, "py");
            int idxPz = Array.IndexOf(cols, "pz");
            int idxRoll = Array.IndexOf(cols, "roll");
            int idxPitch = Array.IndexOf(cols, "pitch");
            int idxYaw = Array.IndexOf(cols, "yaw");

            if (idxT < 0 || idxPx < 0 || idxPy < 0 || idxPz < 0 || idxRoll < 0 || idxPitch < 0 || idxYaw < 0)
                throw new InvalidDataException("Missing required columns in log file");

            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var parts = line.Split(',');
                if (parts.Length <= Math.Max(idxYaw, idxPz))
                    continue;

                float t = ParseFloat(parts[idxT]);
                float px = ParseFloat(parts[idxPx]);
                float py = ParseFloat(parts[idxPy]);
                float pz = ParseFloat(parts[idxPz]);
                float roll = ParseFloat(parts[idxRoll]);
                float pitch = ParseFloat(parts[idxPitch]);
                float yaw = ParseFloat(parts[idxYaw]);

                Vector3 posUnity = new Vector3(px, pz, py);
                Quaternion qSim = Quaternion.Euler(
                    Mathf.Rad2Deg * roll,
                    Mathf.Rad2Deg * pitch,
                    Mathf.Rad2Deg * yaw
                );
                Quaternion qUnity = ConvertSimToUnityRotation(qSim);

                _frames.Add(new Frame { t = t, pos = posUnity, rot = qUnity });
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[LogReplayPlayer] Failed to load log: {e}");
            _frames.Clear();
            return false;
        }

        if (_frames.Count == 0)
        {
            Debug.LogWarning("[LogReplayPlayer] Log has no frames.");
            return false;
        }

        CurrentLogPath = path;
        _startTime = _frames[0].t;
        ApplyFrame(_frames[0]);

        if (playOnLoad)
            Play();
        else
            Pause();

        return true;
    }

    public void Play()
    {
        if (_frames.Count == 0)
            return;
        IsPlaying = true;
    }

    public void Pause()
    {
        IsPlaying = false;
    }

    public void Stop()
    {
        IsPlaying = false;
        _frameIndex = 0;
        _playbackTime = 0f;
        if (_frames.Count > 0)
            ApplyFrame(_frames[0]);
    }

    public void Replay()
    {
        _frameIndex = 0;
        _playbackTime = 0f;
        if (_frames.Count > 0)
            ApplyFrame(_frames[0]);
        Play();
    }

    private void Update()
    {
        if (!IsPlaying || _frames.Count == 0)
            return;

        _playbackTime += Time.deltaTime * timeScale;
        float targetTime = _startTime + _playbackTime;

        while (_frameIndex < _frames.Count - 1 && _frames[_frameIndex + 1].t <= targetTime)
        {
            _frameIndex++;
        }

        ApplyFrame(_frames[_frameIndex]);

        if (_frameIndex >= _frames.Count - 1)
        {
            if (loop)
            {
                Replay();
            }
            else
            {
                Pause();
            }
        }
    }

    private void ApplyFrame(Frame frame)
    {
        if (target == null)
            return;
        target.position = frame.pos;
        target.rotation = frame.rot;
    }

    private static float ParseFloat(string value)
    {
        return float.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    private static Quaternion ConvertSimToUnityRotation(Quaternion qSim)
    {
        Matrix4x4 Rsim = Matrix4x4.Rotate(qSim);
        Matrix4x4 M = Matrix4x4.identity;
        M.m00 = 1; M.m01 = 0; M.m02 = 0;
        M.m10 = 0; M.m11 = 0; M.m12 = 1;
        M.m20 = 0; M.m21 = 1; M.m22 = 0;
        Matrix4x4 Minv = M.transpose;
        Matrix4x4 Runity = M * Rsim * Minv;
        return Runity.rotation;
    }
}
