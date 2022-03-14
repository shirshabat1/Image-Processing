using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

public class Chain : MonoBehaviour
{
    private BezierCurve curve; // The Bezier curve around which to build the chain
    private List<GameObject> chainLinks = new List<GameObject>(); // A list to contain the chain links GameObjects

    public GameObject ChainLink; // Reference to a GameObject representing a chain link
    public float LinkSize = 2.0f; // Distance between links

    // Awake is called when the script instance is being loaded
    public void Awake()
    {
        curve = GetComponent<BezierCurve>();
    }

    // Constructs a chain made of links along the given Bezier curve, updates them in the chainLinks List
    public void ShowChain()
    {
        // Clean up the list of old chain links
        foreach (GameObject link in chainLinks)
        {
            Destroy(link);
        }
        // Counter for alternating the up direction of the chain, even is for the direction of the normal, odd is for the direction of the binormal
        int j = 0;
        float curveLength = curve.ArcLength();
        for (float i = 0; i < curveLength; i += LinkSize)
        {
            float t = curve.ArcLengthToT(i);
            Vector3 position = curve.GetPoint(t);
            Vector3 upDirection = (j % 2 == 0) ? curve.GetNormal(t) : curve.GetBinormal(t);
            chainLinks.Add(CreateChainLink(position, curve.GetTangent(t), upDirection));
            // Increment the counter for deciding up direction
            j++;
        }
    }

    // Instantiates & returns a ChainLink at given position, oriented towards the given forward and up vectors
    public GameObject CreateChainLink(Vector3 position, Vector3 forward, Vector3 up)
    {
        GameObject chainLink = Instantiate(ChainLink);
        chainLink.transform.position = position;
        chainLink.transform.rotation = Quaternion.LookRotation(forward, up);
        chainLink.transform.parent = transform;
        return chainLink;
    }

    // Rebuild chain when BezierCurve component is changed
    public void CurveUpdated()
    {
        ShowChain();
    }
}

[CustomEditor(typeof(Chain))]
class ChainEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        if (GUILayout.Button("Show Chain"))
        {
            var chain = target as Chain;
            chain.ShowChain();
        }
    }
}