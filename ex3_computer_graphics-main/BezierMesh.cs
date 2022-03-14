using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

[ExecuteInEditMode]
public class BezierMesh : MonoBehaviour
{
    private BezierCurve curve; // The Bezier curve around which to build the mesh

    public float Radius = 0.5f; // The distance of mesh vertices from the curve
    public int NumSteps = 16; // Number of points along the curve to sample
    public int NumSides = 8; // Number of vertices created at each point

    private static readonly float FullCircleDegrees = 360;

    // Awake is called when the script instance is being loaded
    public void Awake()
    {
        curve = GetComponent<BezierCurve>();
        BuildMesh();
    }

    // Returns a list of unit directions on the unit circle , i.e. (x,y) vectors , calculated according to the given number of sides
    private static List<Vector2> GetUnitDirection(int numSides)
    {
        List<Vector2> unitDirections = new List<Vector2>();
        float degreeStep = FullCircleDegrees / (float)numSides;
        for (float i = 0; i < FullCircleDegrees; i += degreeStep)
        {
            unitDirections.Add(GetUnitCirclePoint(i));
        }
        return unitDirections;
    }

    private static List<Vector3> GetBezierMeshVertices(BezierCurve curve, float radius, int numSteps, Dictionary<int, List<int>> samplePointToVertices, List<Vector2> unitDirections)
    {
        // Create all vertices for the bezier mesh
        List<Vector3> vertices = new List<Vector3>();
        for (int i = 0; i < numSteps + 1; i++)
        {
            // Sample point and calculate the point's binormal and normal
            float t_i = (float)i / (float)numSteps;
            Vector3 samplePoint = curve.GetPoint(t_i);
            Vector3 b = curve.GetBinormal(t_i);
            Vector3 n = curve.GetNormal(t_i);
            // Add to the sample point's index all the new vertices around it
            if (!samplePointToVertices.ContainsKey(i))
            {
                samplePointToVertices[i] = new List<int>();
            }
            foreach (Vector3 direction in unitDirections)
            {
                // Calculate the vertex position in the direction inside the plane spanned by the binormal and normal vectors
                Vector3 planeDirection = direction.x * b + direction.y * n;
                Vector3 vertex = samplePoint + planeDirection * radius;
                samplePointToVertices[i].Add(vertices.Count);
                vertices.Add(vertex);
            }
        }
        return vertices;
    }

    private static List<Vector4> GetBezierMeshFaces(int numSides, Dictionary<int, List<int>> samplePointToVertices)
    {
        List<Vector4> faces = new List<Vector4>();
        foreach (KeyValuePair<int, List<int>> kvp in samplePointToVertices)
        {
            int samplePoint1 = kvp.Key;
            List<int> samplePoint1Vertices = kvp.Value;
            // Check if there is a sample point following the current one, and if so, connect the matching vertices
            if (samplePointToVertices.ContainsKey(samplePoint1 + 1))
            {
                // Get the next sample point surrounding verices
                int samplePoint2 = samplePoint1 + 1;
                List<int> samplePoint2Vertices = samplePointToVertices[samplePoint2];
                for (int i = 0; i < numSides; i++)
                {
                    int p1 = samplePoint1Vertices[i];
                    int p2 = samplePoint2Vertices[i];
                    int p3 = samplePoint2Vertices[(i + 1) % numSides];
                    int p4 = samplePoint1Vertices[(i + 1) % numSides];
                    Vector4 face = new Vector4(p1, p2, p3, p4);
                    faces.Add(face);
                }
            }
        }
        return faces;
    }

    // Returns a "tube" Mesh built around the given Bézier curve
    public static Mesh GetBezierMesh(BezierCurve curve, float radius, int numSteps, int numSides)
    {
        QuadMeshData meshData = new QuadMeshData();
        Dictionary<int, List<int>> samplePointToVertices = new Dictionary<int, List<int>>();
        List<Vector2> unitDirections = GetUnitDirection(numSides);
        meshData.vertices = GetBezierMeshVertices(curve, radius, numSteps, samplePointToVertices, unitDirections);
        meshData.quads = GetBezierMeshFaces(numSides, samplePointToVertices);
        return meshData.ToUnityMesh();
    }


    // Returns 2D coordinates of a point on the unit circle at a given angle from the x-axis
    private static Vector2 GetUnitCirclePoint(float degrees)
    {
        float radians = degrees * Mathf.Deg2Rad;
        return new Vector2(Mathf.Sin(radians), Mathf.Cos(radians));
    }

    public void BuildMesh()
    {
        var meshFilter = GetComponent<MeshFilter>();
        meshFilter.mesh = GetBezierMesh(curve, Radius, NumSteps, NumSides);
    }

    // Rebuild mesh when BezierCurve component is changed
    public void CurveUpdated()
    {
        BuildMesh();
    }
}



[CustomEditor(typeof(BezierMesh))]
class BezierMeshEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        if (GUILayout.Button("Update Mesh"))
        {
            var bezierMesh = target as BezierMesh;
            bezierMesh.BuildMesh();
        }
    }
}