using System.Collections.Generic;
using System.Linq;
using UnityEngine;


public class BezierCurve : MonoBehaviour
{
    // Bezier control points
    public Vector3 p0;
    public Vector3 p1;
    public Vector3 p2;
    public Vector3 p3;

    private float[] cumLengths; // Cumulative lengths lookup table
    private readonly int numSteps = 128; // Number of points to sample for the cumLengths LUT

    // Returns position B(t) on the Bezier curve for given parameter 0 <= t <= 1
    public Vector3 GetPoint(float t)
    {
        return Mathf.Pow(1 - t, 3) * p0 + 3 * Mathf.Pow(1 - t, 2) * t * p1 + 3 * (1 - t) * Mathf.Pow(t, 2) * p2 + Mathf.Pow(t, 3) * p3;
    }

    // Returns first derivative B'(t) for given parameter 0 <= t <= 1
    public Vector3 GetFirstDerivative(float t)
    {
        return 3 * Mathf.Pow(1 - t, 2) * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * Mathf.Pow(t, 2) * (p3 - p2);
    }

    // Returns second derivative B''(t) for given parameter 0 <= t <= 1
    public Vector3 GetSecondDerivative(float t)
    {
        return 6 * (1 - t) * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1);
    }

    // Returns the tangent vector to the curve at point B(t) for a given 0 <= t <= 1
    public Vector3 GetTangent(float t)
    {
        return GetFirstDerivative(t).normalized;
    }

    // Returns the Frenet normal to the curve at point B(t) for a given 0 <= t <= 1
    public Vector3 GetNormal(float t)
    {
        return Vector3.Cross(GetTangent(t), GetBinormal(t)).normalized;
    }

    // Returns the Frenet binormal to the curve at point B(t) for a given 0 <= t <= 1
    public Vector3 GetBinormal(float t)
    {
        Vector3 v = (GetFirstDerivative(t) + GetSecondDerivative(t)).normalized;
        return Vector3.Cross(GetTangent(t), v).normalized;

    }

    // Calculates the arc-lengths lookup table
    public void CalcCumLengths()
    {
        cumLengths = new float[numSteps + 1];
        Vector3 prevSamplePoint = GetPoint(0);
        for (int i = 1; i < numSteps + 1; i++)
        {
            Vector3 currSamplePoint = GetPoint((float)i / (float)numSteps);
            cumLengths[i] = (currSamplePoint - prevSamplePoint).magnitude + cumLengths[i - 1];
            // Update the previous sample point to avoid double sampling and redundant calculating
            prevSamplePoint = currSamplePoint;
        }
    }

    // Returns the total arc-length of the Bezier curve
    public float ArcLength()
    {
        return cumLengths[numSteps];
    }

    // Returns approximate t s.t. the arc-length to B(t) = arcLength
    public float ArcLengthToT(float a)
    {
        float t = 0;
        for (int i = 0; i < numSteps + 1; i++)
        {
            float t_i = (float)i / (float)numSteps;
            // If we found the exact value, no need to interpolate
            if (cumLengths[i] == a)
            {
                return t_i;
            }
            // If the approximate value is greater that the given one, we are certainly in a positive index with previous value existing
            // Interploate to find the T value
            if (cumLengths[i] > a)
            {
                float t_i_prev = (float)(i - 1) / (float)numSteps;
                // Reverse interpolate to get the interpolation value betweeen the lengths
                float interpVal = Mathf.InverseLerp(a: cumLengths[i - 1], b: cumLengths[i], value: a);
                // Interpolate the T values with the interpolation value to return the final result
                t = Mathf.Lerp(a: t_i_prev, b: t_i, t: interpVal);
                break;
            }
        }
        return t; //in case we reached the end of the LUT, return the maximum value.
    }

    // Start is called before the first frame update
    public void Start()
    {
        Vector3 v1 = GetPoint(0.5f);
        Debug.DrawLine(Vector3.zero, v1, Color.red, 10f);
        Refresh();
    }

    // Update the curve and send a message to other components on the GameObject
    public void Refresh()
    {
        CalcCumLengths();
        if (Application.isPlaying)
        {
            SendMessage("CurveUpdated", SendMessageOptions.DontRequireReceiver);
        }
    }

    // Set default values in editor
    public void Reset()
    {
        p0 = new Vector3(1f, 0f, 1f);
        p1 = new Vector3(1f, 0f, -1f);
        p2 = new Vector3(-1f, 0f, -1f);
        p3 = new Vector3(-1f, 0f, 1f);

        Refresh();
    }
}



