from src.solid_buckling_convergence import _mesh_label


def test_mesh_label_is_filesystem_safe():
    assert _mesh_label(1.2) == "h1p2"
    assert _mesh_label(0.7) == "h0p7"
