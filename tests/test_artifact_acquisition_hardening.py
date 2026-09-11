from tools.acquire_insightface_v07_models import locked_member, safe_member


def test_safe_member_rejects_traversal_and_absolute_paths():
    assert safe_member("buffalo_l/det_10g.onnx")
    assert not safe_member("../det_10g.onnx")
    assert not safe_member("/tmp/det_10g.onnx")
    assert not safe_member("buffalo_l/../det_10g.onnx")
    assert not safe_member("buffalo_l\\det_10g.onnx")


def test_locked_member_rejects_ambiguous_duplicates():
    assert locked_member(["buffalo_l/det_10g.onnx"], "det_10g.onnx") == "buffalo_l/det_10g.onnx"
    try:
        locked_member(["det_10g.onnx", "buffalo_l/det_10g.onnx"], "det_10g.onnx")
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous duplicate locked model must fail closed")
