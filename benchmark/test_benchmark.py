import unittest

from evaluate import compare_episode
from hybrid_common import build_candidates, frames_from_intervals, intervals_from_frames
from normalize import normalize_ours, normalize_sgego, normalize_svg2
from report_hybrid_evaluation import interval_iou
from run_vlm_evaluation import (
    is_local_endpoint, messages_for_model, model_request_args, verifier_visual_plan,
)


class NormalizationTests(unittest.TestCase):
    def test_ours_drops_irrelevant_nodes_and_edges(self):
        source = {
            "frame_count": 1,
            "frames": [{
                "frame_index": 0,
                "nodes": [
                    {"entity_id": "robot", "role": "robot", "status": "visible"},
                    {"entity_id": "cup", "role": "distractor", "status": "visible"},
                ],
                "state_edges": [
                    {"subject": "robot", "relation": "holding", "object": "manipulated_object"},
                    {"subject": "robot", "relation": "near", "object": "cup"},
                ],
                "actions": [],
            }],
        }
        result = normalize_ours(source)
        self.assertEqual(result["frames"][0]["nodes"], ["robot"])
        self.assertEqual(result["frames"][0]["edges"], [["robot", "holding", "manipulated_object"]])

    def test_sgego_expands_window_and_uses_roles(self):
        source = {"0-2": {
            "obj": ["arm", "block", "wall"],
            "role": ["robot", "manipulated_object", None],
            "pair": [[0, 1], [0, 2]],
            "rel": ["Holding", "near"],
        }}
        result = normalize_sgego(source, 3)
        self.assertEqual(result["frames"][0]["edges"], [["robot", "holding", "manipulated_object"]])
        self.assertEqual(result["frames"][1]["nodes"], ["manipulated_object", "robot"])
        self.assertEqual(result["frames"][2]["nodes"], [])

    def test_svg2_only_expands_selected_role_relations(self):
        source = {
            "total_frames": 4,
            "objects": [
                {"object_id": 4, "role": "robot", "trajectory": {"frames": [0, 1, 2, 3]}},
                {"object_id": 8, "role": "target", "trajectory": {"frames": [1, 2]}},
                {"object_id": 9, "trajectory": {"frames": [0, 1, 2, 3]}},
            ],
            "relationships": {
                "sampled_frame_indices": [0, 2, 3],
                "temporal": [[4, "moves toward", 8, [[0, 1]], "motion"], [4, "near", 9, [[0, 2]], "motion"]],
                "spatial": [],
            },
        }
        result = normalize_svg2(source)
        self.assertEqual(result["frames"][2]["edges"], [["robot", "moves toward", "target"]])
        self.assertEqual(result["frames"][3]["edges"], [])


class MetricTests(unittest.TestCase):
    def test_micro_precision_recall(self):
        reference = {"frame_count": 1, "frames": [{
            "frame_index": 0,
            "nodes": ["robot", "target"],
            "edges": [["robot", "touching", "target"]],
        }]}
        prediction = {"frame_count": 1, "frames": [{
            "frame_index": 0,
            "nodes": ["robot", "manipulated_object"],
            "edges": [["robot", "near", "target"]],
        }]}
        result = compare_episode(prediction, reference)
        self.assertEqual(result["nodes"]["precision"], 0.5)
        self.assertEqual(result["nodes"]["recall"], 0.5)
        self.assertEqual(result["triplets"]["precision"], 0.0)
        self.assertEqual(result["triplets"]["recall"], 0.0)


class HybridEvaluationTests(unittest.TestCase):
    def test_local_judge_endpoint_needs_no_real_api_key(self):
        self.assertTrue(is_local_endpoint("http://gemma-judge:8000/v1"))
        self.assertTrue(is_local_endpoint("http://127.0.0.1:8000/v1"))
        self.assertFalse(is_local_endpoint("https://provider.example/v1"))

    def test_gemma4_requests_deterministic_json_without_thinking(self):
        args = model_request_args("google/gemma-4-31B-it")
        self.assertEqual(args["temperature"], 0.0)
        self.assertEqual(args["response_format"], {"type": "json_object"})
        self.assertFalse(args["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_gemma4_places_images_before_text(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "prompt"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,x"}},
        ]}]
        ordered = messages_for_model("google/gemma-4-31B-it", messages)
        self.assertEqual(ordered[0]["content"][0]["type"], "image_url")
        self.assertIs(messages_for_model("Qwen/test", messages), messages)

    def test_interval_round_trip(self):
        intervals = intervals_from_frames([0, 1, 4, 5, 6])
        self.assertEqual(intervals, [[0, 1], [4, 6]])
        self.assertEqual(frames_from_intervals(intervals, 7), {0, 1, 4, 5, 6})

    def test_candidate_union_keeps_sources(self):
        output = {
            "role_labels": {"robot": "robot arm"},
            "role_intervals": {"robot": [[0, 2]]},
            "relations": [{
                "subject": "robot", "predicate": "holding",
                "object": "manipulated_object", "intervals": [[1, 2]],
            }],
        }
        candidates = build_candidates({"ours": output, "sg_ego": output, "svg2": None})
        roles = [fact for fact in candidates["facts"] if fact["kind"] == "role"]
        self.assertEqual(len(roles), 2)
        self.assertEqual(
            {tuple(fact["sources"]) for fact in roles},
            {("ours",), ("sg_ego",)},
        )
        self.assertEqual(
            candidates["method_facts"]["ours"]["roles"],
            candidates["method_facts"]["sg_ego"]["roles"],
        )
        self.assertEqual(len(candidates["method_facts"]["ours"]["relations"]), 1)

    def test_temporal_iou(self):
        self.assertAlmostEqual(interval_iou([[0, 2]], [[1, 3]]), 0.5)

    def test_verifier_frames_include_track_evidence(self):
        facts = [{
            "id": "r_track", "kind": "role", "role": "robot", "label": "arm",
            "evidence_boxes": [{"frame_index": 7, "bbox": [0, 0, 10, 10]}],
        }]
        indices, overlays = verifier_visual_plan(facts, frame_count=10, maximum=6)
        self.assertIn(7, indices)
        self.assertEqual(overlays[7][0][0], "r_track")


if __name__ == "__main__":
    unittest.main()
