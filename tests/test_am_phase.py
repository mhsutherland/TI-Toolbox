"""Host tests for ``tit.tools.am_phase``: which simulations it accepts, and its CLI.

The physics is checked against real numpy in ``tests/numerical/test_am_phase.py``.
"""

import json
from unittest.mock import patch

import pytest

from tit.tools import am_phase

_TI_CONFIG = {
    "simulation_mode": "TI",
    "electrode_pairs": [["E1", "E2"], ["E3", "E4"]],
    "conductivity": "scalar",
    "intensities": [2.0, 2.0],
    "carrier_only": False,
}


def test_two_pair_ti_is_accepted():
    am_phase.check_simulation_config(dict(_TI_CONFIG))


@pytest.mark.parametrize(
    "change",
    [
        {"simulation_mode": "mTI"},
        {"electrode_pairs": [["E1", "E2"]] * 4},
        {"carrier_only": True},
    ],
)
def test_other_simulations_are_refused(change):
    with pytest.raises(ValueError):
        am_phase.check_simulation_config({**_TI_CONFIG, **change})


def test_output_paths_list_the_json_first(tmp_path):
    paths = am_phase.output_paths(str(tmp_path), "S", skip_nifti=False)
    assert paths[0].endswith("S_AMphase.json")
    assert len(paths) == 7
    assert am_phase.output_paths(str(tmp_path), "S", skip_nifti=True) == paths[:1]


class _FakePM:
    def __init__(self, root):
        self.root = root

    def m2m(self, sid):
        return str(self.root / f"m2m_{sid}")

    def simulation(self, sid, sim):
        return str(self.root / "Simulations" / sim)

    def t1(self, sid):
        return str(self.root / f"m2m_{sid}" / "T1.nii.gz")

    def tissue_labeling(self, sid):
        return str(self.root / f"m2m_{sid}" / "segmentation" / "labeling.nii.gz")


def _simulation(tmp_path, config, *, inputs=True):
    sim_dir = tmp_path / "Simulations" / "S"
    (sim_dir / "documentation").mkdir(parents=True)
    (sim_dir / "TI" / "niftis").mkdir(parents=True)
    (sim_dir / "documentation" / "config.json").write_text(json.dumps(config))
    if inputs:
        (sim_dir / "high_Frequency" / "mesh").mkdir(parents=True)
        for i in (1, 2):
            (sim_dir / "high_Frequency" / "mesh" / f"X_TDCS_{i}_scalar.msh").touch()
        (tmp_path / "m2m_X").mkdir()
        (tmp_path / "m2m_X" / "T1.nii.gz").touch()
    return sim_dir


def test_a_missing_simulation_is_named(tmp_path):
    with patch("tit.paths.get_path_manager", return_value=_FakePM(tmp_path)):
        with pytest.raises(FileNotFoundError, match="No simulation 'S'"):
            am_phase.run_am_phase("X", "S")


def test_missing_meshes_are_listed_before_anything_is_read(tmp_path):
    _simulation(tmp_path, _TI_CONFIG, inputs=False)
    with patch("tit.paths.get_path_manager", return_value=_FakePM(tmp_path)):
        with pytest.raises(FileNotFoundError, match="X_TDCS_1_scalar.msh"):
            am_phase.run_am_phase("X", "S")


def test_existing_outputs_stop_the_run_before_any_mesh_is_read(tmp_path):
    sim_dir = _simulation(tmp_path, _TI_CONFIG)
    (sim_dir / "TI" / "niftis" / "S_AMphase.json").write_text("{}")
    with patch("tit.paths.get_path_manager", return_value=_FakePM(tmp_path)):
        with pytest.raises(FileExistsError, match="--overwrite"):
            am_phase.run_am_phase("X", "S")


def test_an_mti_simulation_is_refused_before_any_mesh_is_read(tmp_path):
    _simulation(tmp_path, {**_TI_CONFIG, "simulation_mode": "mTI"})
    with patch("tit.paths.get_path_manager", return_value=_FakePM(tmp_path)):
        with pytest.raises(ValueError, match="two-pair TI"):
            am_phase.run_am_phase("X", "S")


def test_cli_passes_its_options_through():
    report = {"report_path": "/r/S_AMphase.json", "niftis": {}}
    with (
        patch.object(am_phase, "run_am_phase", return_value=report) as run,
        patch.object(am_phase, "_report_job_outputs") as announce,
    ):
        am_phase.main(
            ["--subject", "X", "--sim", "S", "--overwrite", "--roi-label", "53"]
        )
    run.assert_called_once_with(
        "X",
        "S",
        project=None,
        overwrite=True,
        skip_nifti=False,
        n_check=2000,
        roi_label=53,
    )
    announce.assert_called_once_with(report, "/r/S_AMphase.json")


def test_cli_turns_a_refusal_into_a_clean_exit():
    with patch.object(
        am_phase, "run_am_phase", side_effect=FileExistsError("Outputs exist")
    ):
        with pytest.raises(SystemExit, match="Outputs exist"):
            am_phase.main(["--subject", "X", "--sim", "S"])
