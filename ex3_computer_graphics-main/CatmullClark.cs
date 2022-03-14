using System;
using System.Collections.Generic;
using UnityEngine;


public class CCMeshData
{
    public List<Vector3> points; // Original mesh points
    public List<Vector4> faces; // Original mesh quad faces
    public List<Vector4> edges; // Original mesh edges
    public List<Vector3> facePoints; // Face points, as described in the Catmull-Clark algorithm
    public List<Vector3> edgePoints; // Edge points, as described in the Catmull-Clark algorithm
    public List<Vector3> newPoints; // New locations of the original mesh points, according to Catmull-Clark
}

public class Vec2Comperar : EqualityComparer<Vector2>
{
    private static readonly float EPSILON = 1e-5f;

    public override bool Equals(Vector2 v1, Vector2 v2)
    {
        // We use Vector2 for edges which can appear in the reverse order
        // Since we want unique key for every edge, we also compare when one of the vectors is flipped
        Vector2 v1Flipped = new Vector2(v1.y, v1.x);
        return (Vector2.Distance(v1, v2) < EPSILON) || (Vector2.Distance(v1Flipped, v2) < EPSILON);
    }

    public override int GetHashCode(Vector2 v)
    {
        return 0;
    }
}


public static class CatmullClark
{
    // Returns a QuadMeshData representing the input mesh after one iteration of Catmull-Clark subdivision.
    public static QuadMeshData Subdivide(QuadMeshData quadMeshData)
    {
        // Create and initialize a CCMeshData corresponding to the given QuadMeshData
        CCMeshData meshData = new CCMeshData();
        meshData.points = quadMeshData.vertices;
        meshData.faces = quadMeshData.quads;
        meshData.edges = GetEdges(meshData);
        meshData.facePoints = GetFacePoints(meshData);
        meshData.edgePoints = GetEdgePoints(meshData);
        meshData.newPoints = GetNewPoints(meshData);

        //Combine facePoints, edgePoints and newPoints into a subdivided QuadMeshData
        // Create a list for all the points of the new mesh, i.e. the new points, the edge points and the face points
        List<Vector3> newMeshPoints = new List<Vector3>(meshData.newPoints);
        // Save the starting index of the edge points and add them 
        int edgePointsStartIdx = newMeshPoints.Count;
        newMeshPoints.AddRange(meshData.edgePoints);
        // Save the starting index of the face points and add them
        int facePointsStartIdx = newMeshPoints.Count;
        newMeshPoints.AddRange(meshData.facePoints);
        // newMeshPoints now contains [newPoints, edgePoints, facePoints] with the starting indices for every interval saved
        // And the original mesh points indices matching the indices of the new points

        // Create a dictionary mapping each edge vertices to index of the edge in the meshData edges list
        Dictionary<Vector2, int> pointsToEdgeIdx = GetPointsToEdgeIdxDict(meshData);

        // Iterate over every face the create the new 4 faces from it
        List<Vector4> newMeshFaces = CreateNewFaces(meshData, pointsToEdgeIdx, edgePointsStartIdx, facePointsStartIdx);

        return new QuadMeshData(vertices: newMeshPoints, quads: newMeshFaces);
    }


    private static Dictionary<Vector2, int> GetPointsToEdgeIdxDict(CCMeshData mesh)
    {
        Dictionary<Vector2, int> pointsToEdgeIdx = new Dictionary<Vector2, int>(new Vec2Comperar());
        for (int i = 0; i < mesh.edges.Count; i++)
        {
            // Extract the points from the edge vector and map it to the edge's original index
            Vector4 edge = mesh.edges[i];
            Vector2 edgePointsVec = new Vector2(edge[0], edge[1]);
            pointsToEdgeIdx[edgePointsVec] = i;
        }
        return pointsToEdgeIdx;
    }

    private static List<Vector4> CreateNewFaces(CCMeshData mesh, Dictionary<Vector2, int> pointsToEdgeIdx, int edgePointsStartIdx, int facePointsStartIdx)
    {
        List<Vector4> newMeshFaces = new List<Vector4>();
        for (int i = 0; i < mesh.faces.Count; i++)
        {
            Vector4 face = mesh.faces[i];
            int facePointIdx = facePointsStartIdx + i;
            for (int j = 0; j < 4; j++)
            {
                int point = (int)face[j];
                int adjacentPoint1 = (int)face[(j + 3) % 4];
                int adjacentPoint2 = (int)face[(j + 1) % 4];
                int edgePoint1 = edgePointsStartIdx + pointsToEdgeIdx[new Vector2(adjacentPoint1, point)];
                int edgePoint2 = edgePointsStartIdx + pointsToEdgeIdx[new Vector2(point, adjacentPoint2)];

                Vector4 newFace = new Vector4(facePointIdx, edgePoint1, point, edgePoint2);
                newMeshFaces.Add(newFace);
            }
        }
        return newMeshFaces;
    }


    // Returns a list of all edges in the mesh defined by given points and faces.
    // Each edge is represented by Vector4(p1, p2, f1, f2)
    // p1, p2 are the edge vertices
    // f1, f2 are faces incident to the edge. If the edge belongs to one face only, f2 is -1
    public static List<Vector4> GetEdges(CCMeshData mesh)
    {
        // Create a mapping between the edge's points vertices to the edge itself
        Dictionary<Vector2, Vector4> dict = new Dictionary<Vector2, Vector4>(new Vec2Comperar());
        for (int j = 0; j < mesh.faces.Count; j++)
        {
            Vector4 f = mesh.faces[j];
            for (int i = 0; i < 4; i++)
            {
                float p1 = f[i];
                float p2 = f[(i + 1) % 4]; // So that p4 will connect to p1 in the final face's edge
                Vector2 edge = new Vector2(p1, p2);
                if (dict.ContainsKey(edge))
                {
                    // If the edge already exist, we saw it in a previous face (f1), connect it to f2
                    Vector4 currUniqueEdge = dict[edge];
                    dict[edge] = new Vector4(currUniqueEdge[0], currUniqueEdge[1], currUniqueEdge[2], j);
                }
                else
                {
                    // First time seeing the edge, connect it the f1 (j) and keep f2 as -1
                    dict[edge] = new Vector4(p1, p2, j, -1);
                }
            }
        }
        return new List<Vector4>(dict.Values);
    }

    // Returns a list of "face points" for the given CCMeshData, as described in the Catmull-Clark algorithm 
    public static List<Vector3> GetFacePoints(CCMeshData mesh)
    {
        List<Vector3> facePoints = new List<Vector3>();
        foreach (Vector4 f in mesh.faces)
        {
            Vector3 facePoint = (mesh.points[(int)f[0]] + mesh.points[(int)f[1]] + mesh.points[(int)f[2]] + mesh.points[(int)f[3]]) / 4;
            facePoints.Add(facePoint);
        }
        return facePoints;
    }

    // Returns a list of "edge points" for the given CCMeshData, as described in the Catmull-Clark algorithm 
    public static List<Vector3> GetEdgePoints(CCMeshData mesh)
    {
        List<Vector3> edgePoints = new List<Vector3>();
        foreach (Vector4 edge in mesh.edges)
        {
            Vector3 p1 = mesh.points[(int)edge[0]];
            Vector3 p2 = mesh.points[(int)edge[1]];
            int f1 = (int)edge[2];
            int f2 = (int)edge[3];
            Vector3 facePoint1 = mesh.facePoints[f1];
            if (edge[3] != -1)
            {
                Vector3 facePoint2 = mesh.facePoints[f2];
                edgePoints.Add((facePoint1 + facePoint2 + p1 + p2) / 4);
            }
            else
            {
                edgePoints.Add((facePoint1 + p1 + p2) / 3);
            }
        }
        return edgePoints;
    }

    // Returns a list of new locations of the original points for the given CCMeshData, as described in the CC algorithm 
    public static List<Vector3> GetNewPoints(CCMeshData mesh)
    {
        List<Vector3> newPoints = new List<Vector3>();
        Dictionary<int, List<Vector3>> midPointsDict = new Dictionary<int, List<Vector3>>();
        Dictionary<int, List<Vector3>> facePointDict = new Dictionary<int, List<Vector3>>();

        // Create a dictionary mapping each point index to a list of all its adjacent edge points
        foreach (Vector4 edge in mesh.edges)
        {
            int p1Index = (int)edge[0];
            int p2Index = (int)edge[1];
            Vector3 midPoint = (mesh.points[p1Index] + mesh.points[p2Index]) / 2;
            InsertNewPoint(p1Index, midPointsDict, midPoint);
            InsertNewPoint(p2Index, midPointsDict, midPoint);
        }

        // Create a dictionary mapping each point index to a list of all its adjacent edge points
        for (int i = 0; i < mesh.faces.Count; i++)
        {
            Vector4 face = mesh.faces[i];
            Vector3 facePoint = mesh.facePoints[i];
            // Add the face's face poit for every point in the face
            for (int j = 0; j < 4; j++)
            {
                int p = (int)face[j]; // Index of the point
                InsertNewPoint(p, facePointDict, facePoint);
            }
        }

        // Calculate the final position for every original point
        for (int i = 0; i < mesh.points.Count; i++)
        {
            Vector3 r = Vector3.zero;
            List<Vector3> midPoints = midPointsDict[i];
            foreach (Vector3 midPoint in midPoints)
            {
                r += midPoint;
            }
            r /= midPoints.Count;
            Vector3 f = Vector3.zero;
            List<Vector3> facePoints = facePointDict[i];
            foreach (Vector3 facePoint in facePoints)
            {
                f += facePoint;
            }
            f /= facePoints.Count;
            int n = facePoints.Count;
            Vector3 newPos = (f + 2 * r + (n - 3) * mesh.points[i]) / n;
            newPoints.Add(newPos);
        }
        return newPoints;
    }

    private static void InsertNewPoint(int key, Dictionary<int, List<Vector3>> dict, Vector3 point)
    {
        if (!dict.ContainsKey(key))
        {
            dict[key] = new List<Vector3>();
        }
        dict[key].Add(point);
    }
}