#!/usr/bin/env simnibs_python
"""AM-phase maps for a two-pair TI simulation.

Under SimNIBS's quasi-static, purely resistive model each pair's field is a
real vector field times one sinusoid, so at every point

    E(t) = E1 cos(theta) + E2 cos(theta + psi),

with ``theta`` the carrier phase and ``psi`` the carrier-2 minus carrier-1
phase slip, which advances by 2*pi per beat and is the same everywhere in the
head.  The squared envelope of ``|E|`` is convex in ``cos(psi)``, so it peaks
at ``psi = 0`` where ``E1 . E2 > 0`` and at ``psi = pi`` where
``E1 . E2 < 0``, never in between.  The map does not depend on the frequency
offset.  Which side is called ``0`` follows the configured polarity (the first
electrode of each pair is ``+``): reversing one pair swaps the two labels.

What the label is -- and is not
-------------------------------
The label is the phase of the envelope of the field *magnitude* ``|E|``.  Along
a fixed direction ``n`` (a cortical normal, a neuron's preferred axis, the axis
of a recording pair) the envelope peaks at ``psi = 0`` or ``pi`` according to
the sign of ``(n . E1)(n . E2)``, which can differ from the sign of
``E1 . E2`` wherever the two fields are not parallel.  Do not read the label
as the phase a direction-sensitive population or a recording contact sees.

Near 90 degrees between ``E1`` and ``E2`` the two candidate peaks have nearly
equal height (exactly equal at 90 degrees), so the ``|E|`` envelope peaks
twice per beat and the label separates two almost identical cases.  The
modulation depth does not vanish there: for perpendicular fields of equal
magnitude ``a`` it is ``(sqrt(2) - 1) a``.  Treat labels near the
``cos = 0`` border with caution.

The result rests on the resistive model with frequency-independent
conductivities; any tissue phase shift at the carrier frequency, which would
move the peaks away from exactly 0 or pi, is outside it.

Outputs, in ``<simulation>/TI/niftis/``
---------------------------------------
``<sim>_AMphase_label.nii.gz``
    uint8: 0 outside GM+WM, 1 = envelope peaks at psi = 0, 2 = at psi = pi.
``<sim>_AMphase_label_GM.nii.gz``
    The same labels in grey matter only (0 elsewhere).
``<sim>_AMphase_label_LUT.txt``, ``<sim>_AMphase_label_GM_LUT.txt``
    Colour tables for the two label volumes.
``<sim>_AMphase_depth.nii.gz``
    ``|E|``-envelope depth, max minus min over a beat, V/m.  Not ``TI_max``.
``<sim>_AMphase_cos.nii.gz``
    ``cos`` of the angle between ``E1`` and ``E2``.
``<sim>_AMphase.json``
    Definitions, inputs, element volumes, an independent brute-force check
    and the NIfTI checks.  Always written.

All volumes are on the subject's T1 grid (``m2m_<subject>/T1.nii.gz``),
converted with :func:`tit.tools.mesh2nii.msh_to_nifti`.  The label volumes
threshold the interpolated ``cos`` at 0, so a voxel at exactly 90 degrees is
labelled ``pi``.

Usage
-----
Inside the TI-Toolbox container, for a finished two-pair TI simulation::

    simnibs_python -m tit.tools.am_phase --subject <id> --sim <name> \\
        [--project /mnt/<project>] [--roi-label N] [--overwrite]

``--roi-label`` adds a summary of the volume with that value in
``m2m_<subject>/segmentation/labeling.nii.gz`` to the JSON.

See Also
--------
tit.fields.hf_peak : The envelope maximum here, ``max(|E1+E2|, |E1-E2|)``.
tit.calc.get_TI_vectors : The directional modulation envelope (``TI_max``).
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from typing import Any

import numpy as np

from tit import constants as const
from tit.fields import hf_peak

LABEL_PSI0, LABEL_PSIPI = 1, 2
_COL_PSI0 = (0x1B, 0xAF, 0x7A)  # aqua
_COL_PSIPI = (0xED, 0xA1, 0x00)  # amber
_OUTPUT_KEYS = (
    "label.nii.gz",
    "label_LUT.txt",
    "label_GM.nii.gz",
    "label_GM_LUT.txt",
    "depth.nii.gz",
    "cos.nii.gz",
)

DEFINITIONS = {
    "psi": "carrier-2 phase minus carrier-1 phase; advances 2*pi per beat; same everywhere",
    "label": (
        "1 = |E| envelope peaks at psi = 0 (E1.E2 > 0); 2 = peaks at psi = pi "
        "(E1.E2 <= 0); 0 = outside GM+WM. Phase of the field-magnitude envelope, "
        "not of the envelope along any fixed direction."
    ),
    "cos": "E1.E2 / (|E1||E2|)",
    "depth": (
        "max minus min over a beat of the |E| envelope (semi-major axis of the "
        "carrier ellipse), V/m. Does not vanish at 90 degrees: for perpendicular "
        "fields of equal magnitude a it is (sqrt(2) - 1) a, and the envelope then "
        "peaks equally at psi = 0 and pi. Not TI_max."
    ),
    "model": (
        "quasi-static, purely resistive (SimNIBS); independent of the frequency "
        "offset; reversing one pair swaps labels 1 and 2"
    ),
}


# ── Physics ──────────────────────────────────────────────────────────────────


def _lam(aa, bb, ab, c):
    """Squared semi-major axis of the carrier ellipse at ``cos(psi) = c``.

    ``E(theta) = u cos(theta) + v sin(theta)`` with ``u = E1 + E2 c`` and
    ``v = -E2 s``; this is the largest eigenvalue of the Gram matrix of
    ``(u, v)``.
    """
    s2 = 1.0 - c * c
    uu = aa + 2.0 * c * ab + c * c * bb
    vv = s2 * bb
    q = ab + c * bb
    h = 0.5 * (uu - vv)
    return 0.5 * (uu + vv) + np.sqrt(h * h + s2 * q * q)


def am_phase_metrics(E1: np.ndarray, E2: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row AM-phase label, ``cos`` of the field angle, and envelope extremes.

    The squared envelope is the maximum over directions of a function linear
    in ``c = cos(psi)``, hence convex in ``c``: its maximum is at ``c = +-1``
    (``psi = 0`` or ``pi``), equal to :func:`tit.fields.hf_peak`, and its
    minimum is found by golden-section search on ``[-1, 1]``.

    Parameters
    ----------
    E1, E2 : np.ndarray, shape (N, 3)
        Field vectors of pair 1 and pair 2, V/m.

    Returns
    -------
    dict
        ``label`` (uint8, :data:`LABEL_PSI0` or :data:`LABEL_PSIPI`),
        ``cos``, ``env_max``, ``env_min`` and ``depth`` (V/m), each shape (N,).
    """
    aa = np.einsum("ij,ij->i", E1, E1)
    bb = np.einsum("ij,ij->i", E2, E2)
    ab = np.einsum("ij,ij->i", E1, E2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(aa * bb > 0, ab / np.sqrt(aa * bb), 0.0)
    env_max = hf_peak(E1, E2)

    lo = np.full(aa.shape, -1.0)
    hi = np.full(aa.shape, 1.0)
    g = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - g * (hi - lo)
    x2 = lo + g * (hi - lo)
    f1, f2 = _lam(aa, bb, ab, x1), _lam(aa, bb, ab, x2)
    for _ in range(60):
        left = f1 < f2
        hi = np.where(left, x2, hi)
        lo = np.where(left, lo, x1)
        x2n = np.where(left, x1, lo + g * (hi - lo))
        x1n = np.where(left, hi - g * (hi - lo), x2)
        x1, x2 = x1n, x2n
        f1 = _lam(aa, bb, ab, x1)
        f2 = _lam(aa, bb, ab, x2)
    lam_min = np.minimum.reduce(
        [f1, f2, _lam(aa, bb, ab, -1.0 + 0 * aa), _lam(aa, bb, ab, 1.0 + 0 * aa)]
    )
    env_min = np.sqrt(np.maximum(lam_min, 0.0))
    label = np.where(ab > 0, LABEL_PSI0, LABEL_PSIPI).astype(np.uint8)
    return {
        "label": label,
        "cos": cos,
        "env_max": env_max,
        "env_min": env_min,
        "depth": env_max - env_min,
    }


def brute_force_check(E1, E2, metrics, n=2000, seed=0) -> dict:
    """Independent check: sample ``|E1 cos t + E2 cos(t + psi)|`` on a 360x360 grid.

    Elements whose two candidate peaks are within 0.1 % of each other are
    counted as near ties and not checked for the peak location.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(E1), size=min(n, len(E1)), replace=False)
    th = np.deg2rad(np.arange(360))
    ps = np.deg2rad(np.arange(360))
    ct = np.cos(th)[None, :]
    ctp = np.cos(th[None, :] + ps[:, None])  # (psi, theta)
    bad_peak, depth_err, ties = 0, 0.0, 0
    for k in idx:
        a, b = E1[k], E2[k]
        ex = a[0] * ct + b[0] * ctp
        ey = a[1] * ct + b[1] * ctp
        ez = a[2] * ct + b[2] * ctp
        env = np.sqrt(ex * ex + ey * ey + ez * ez).max(axis=1)  # over theta
        e0, epi = env[0], env[180]
        if abs(e0 - epi) < 1e-3 * env.max():
            ties += 1
        else:
            pk = int(np.argmax(env))
            expect = 0 if metrics["label"][k] == LABEL_PSI0 else 180
            if pk != expect:
                bad_peak += 1
        d = env.max() - env.min()
        depth_err = max(
            depth_err, abs(d - metrics["depth"][k]) / max(metrics["env_max"][k], 1e-30)
        )
    return {
        "elements_checked": int(len(idx)),
        "peak_not_at_0_or_180_matching_label": int(bad_peak),
        "near_ties_skipped": int(ties),
        "max_rel_depth_error_vs_grid": float(depth_err),
    }


# ── NIfTI ────────────────────────────────────────────────────────────────────


def save_nifti_gz(data, ref_img, path) -> None:
    """Write *data* on *ref_img*'s grid as ``.nii.gz``.

    A plain ``.nii`` is written to a temporary directory and gzipped with the
    standard library: nibabel's gzip save fails on the container's bind mounts.
    """
    import nibabel as nib

    img = nib.Nifti1Image(data, ref_img.affine)
    img.set_qform(ref_img.get_qform(), code=int(ref_img.header["qform_code"]))
    img.set_sform(ref_img.get_sform(), code=int(ref_img.header["sform_code"]))
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "x.nii")
        nib.save(img, tmp)
        with open(tmp, "rb") as fi, gzip.open(tmp + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        shutil.copy2(tmp + ".gz", path)


def _write_lut(path: str, outside: str) -> None:
    with open(path, "w") as fh:
        fh.write("#No.\t  Label Name:\t\t\t   R   G   B   A\n")
        fh.write(f"0\t  {outside}\t\t\t   0 0 0 0\n")
        fh.write("1\t  AM-peak-at-psi-0\t\t   %d %d %d 255\n" % _COL_PSI0)
        fh.write("2\t  AM-peak-at-psi-pi\t\t   %d %d %d 255\n" % _COL_PSIPI)


def write_niftis(
    mesh,
    metrics: dict,
    m2m_dir: str,
    t1_path: str,
    out_dir: str,
    sim: str,
    labeling_path: str | None = None,
    roi_label: int | None = None,
) -> tuple[dict, dict]:
    """Write the label, depth and ``cos`` volumes and their colour tables.

    The element fields are converted with
    :func:`tit.tools.mesh2nii.msh_to_nifti` (linear interpolation, per tissue)
    into a temporary directory, thresholded and masked, and written beside the
    simulation's other NIfTIs.

    Parameters
    ----------
    mesh : simnibs Msh
        The GM+WM mesh the metrics were computed on; its fields are replaced.
    metrics : dict
        Output of :func:`am_phase_metrics`.
    m2m_dir, t1_path : str
        The subject's head-model directory and T1 (the output grid).
    out_dir, sim : str
        Output directory and the simulation name used in file names.
    labeling_path : str, optional
        Segmentation volume read for the optional ROI summary.
    roi_label : int, optional
        Value in *labeling_path* to summarise; no summary when ``None``.

    Returns
    -------
    paths : dict
        Written file paths by key.
    checks : dict
        Grid, label-value and volume checks, plus the ROI summary if asked.
    """
    import nibabel as nib
    from simnibs import mesh_io

    from tit.tools.mesh2nii import msh_to_nifti

    mesh.elmdata = []
    mesh.nodedata = []
    mesh.add_element_field(metrics["cos"].astype(float), "cos")
    mesh.add_element_field(metrics["depth"].astype(float), "depth")
    mesh.add_element_field(np.ones(mesh.elm.nr), "mask")
    # Interpolation runs per tissue, so gm > 0.5 is "voxel centre in a GM element".
    mesh.add_element_field((mesh.elm.tag1 == const.GM_TISSUE_TAG).astype(float), "gm")
    names = ("cos", "depth", "mask", "gm")
    with tempfile.TemporaryDirectory() as td:
        tmp_msh = os.path.join(td, "am_phase.msh")
        mesh_io.write_msh(mesh, tmp_msh)
        prefix = os.path.join(td, "am")
        msh_to_nifti(tmp_msh, m2m_dir, prefix, fields=list(names))
        vol = {k: nib.load(f"{prefix}_{k}.nii.gz") for k in names}
        arr = {k: np.asarray(v.dataobj, dtype=np.float32) for k, v in vol.items()}
        mask_affine = vol["mask"].affine

    t1 = nib.load(t1_path)
    inside = arr["mask"] > 0.5
    label = np.zeros(inside.shape, np.uint8)
    label[inside & (arr["cos"] > 0)] = LABEL_PSI0
    label[inside & (arr["cos"] <= 0)] = LABEL_PSIPI
    cos = np.where(inside, arr["cos"], 0).astype(np.float32)
    depth = np.where(inside, arr["depth"], 0).astype(np.float32)
    label_gm = np.where(inside & (arr["gm"] > 0.5), label, 0).astype(np.uint8)

    paths = {
        "label": os.path.join(out_dir, f"{sim}_AMphase_label.nii.gz"),
        "label_GM": os.path.join(out_dir, f"{sim}_AMphase_label_GM.nii.gz"),
        "depth": os.path.join(out_dir, f"{sim}_AMphase_depth.nii.gz"),
        "cos": os.path.join(out_dir, f"{sim}_AMphase_cos.nii.gz"),
        "lut": os.path.join(out_dir, f"{sim}_AMphase_label_LUT.txt"),
        "lut_GM": os.path.join(out_dir, f"{sim}_AMphase_label_GM_LUT.txt"),
    }
    save_nifti_gz(label, t1, paths["label"])
    save_nifti_gz(label_gm, t1, paths["label_GM"])
    save_nifti_gz(depth, t1, paths["depth"])
    save_nifti_gz(cos, t1, paths["cos"])
    _write_lut(paths["lut"], "Outside")
    _write_lut(paths["lut_GM"], "Outside-GM")

    vox_cm3 = abs(np.linalg.det(t1.affine[:3, :3])) / 1000.0

    def volumes(lab):
        return {
            "psi0": float((lab == LABEL_PSI0).sum() * vox_cm3),
            "psipi": float((lab == LABEL_PSIPI).sum() * vox_cm3),
        }

    nz = label_gm > 0
    checks: dict[str, Any] = {
        "shape_matches_T1": tuple(label.shape) == tuple(t1.shape[:3]),
        "affine_matches_T1": bool(np.allclose(mask_affine, t1.affine, atol=1e-4)),
        "label_values": sorted(int(v) for v in np.unique(label)),
        "label_GM_values": sorted(int(v) for v in np.unique(label_gm)),
        "label_GM_subset_of_label": bool(np.array_equal(label_gm[nz], label[nz])),
        "volume_cm3": volumes(label),
        "volume_cm3_GM": volumes(label_gm),
    }
    if roi_label is not None and labeling_path:
        checks["roi"] = {
            "labeling": labeling_path,
            "value": int(roi_label),
            "all": roi_summary(labeling_path, roi_label, t1, label, cos, depth),
            "GM": roi_summary(labeling_path, roi_label, t1, label_gm, cos, depth),
        }
    return paths, checks


def roi_summary(labeling_path, value, t1, label, cos, depth) -> dict | None:
    """Phase statistics inside one value of a label volume, matched by world coordinates.

    Returns ``None`` when the label volume is missing or holds no voxel with
    *value*.
    """
    import nibabel as nib

    if not os.path.isfile(labeling_path):
        return None
    lab = nib.load(labeling_path)
    ijk = np.argwhere(np.asarray(lab.dataobj) == value)
    if len(ijk) == 0:
        return None
    world = lab.affine @ np.c_[ijk, np.ones(len(ijk))].T
    t = np.rint(np.linalg.inv(t1.affine) @ world)[:3].astype(int).T
    ok = np.all((t >= 0) & (t < np.array(label.shape)), axis=1)
    t = t[ok]
    L = label[t[:, 0], t[:, 1], t[:, 2]]
    sel = L > 0
    if not sel.any():
        return {"voxels": int(len(t)), "in_mask": 0}
    c = cos[t[:, 0], t[:, 1], t[:, 2]][sel]
    d = depth[t[:, 0], t[:, 1], t[:, 2]][sel]
    return {
        "voxels_in_mask": int(sel.sum()),
        "fraction_psi0": float((L[sel] == LABEL_PSI0).mean()),
        "fraction_psipi": float((L[sel] == LABEL_PSIPI).mean()),
        "median_angle_deg": float(np.degrees(np.arccos(np.clip(np.median(c), -1, 1)))),
        "median_depth_V_per_m": float(np.median(d)),
    }


# ── Driver ───────────────────────────────────────────────────────────────────


def check_simulation_config(cfg: dict) -> None:
    """Refuse simulations this analysis does not apply to.

    Raises
    ------
    ValueError
        If the simulation is not two-pair TI, or is a carrier-only control
        (no beat, so no envelope phase).
    """
    if cfg.get("simulation_mode") != "TI" or len(cfg.get("electrode_pairs", [])) != 2:
        raise ValueError("AM phase handles two-pair TI simulations only.")
    if cfg.get("carrier_only"):
        raise ValueError(
            "Carrier-only (ΔF = 0) simulations have no beat, so no AM phase."
        )


def output_paths(nifti_dir: str, sim: str, skip_nifti: bool) -> list[str]:
    """Every file a run will write, the JSON first."""
    paths = [os.path.join(nifti_dir, f"{sim}_AMphase.json")]
    if not skip_nifti:
        paths += [os.path.join(nifti_dir, f"{sim}_AMphase_{k}") for k in _OUTPUT_KEYS]
    return paths


def run_am_phase(
    subject: str,
    sim: str,
    *,
    project: str | None = None,
    overwrite: bool = False,
    skip_nifti: bool = False,
    n_check: int = 2000,
    roi_label: int | None = None,
) -> dict:
    """Compute and write the AM-phase outputs of one simulation.

    Parameters
    ----------
    subject : str
        Subject ID, without ``sub-``.
    sim : str
        Simulation name (its folder under ``Simulations/``).
    project : str, optional
        Project root; auto-detected inside the container when omitted.
    overwrite : bool, default False
        Replace existing outputs.  Without it a run with any output already
        present stops before computing anything.
    skip_nifti : bool, default False
        Write only the JSON.
    n_check : int, default 2000
        Elements sampled by :func:`brute_force_check`.
    roi_label : int, optional
        Value in the subject's ``segmentation/labeling.nii.gz`` to summarise.

    Returns
    -------
    dict
        The report written to ``<sim>_AMphase.json``, plus ``report_path``
        (where it was written; not itself in the file).

    Raises
    ------
    FileNotFoundError
        If the simulation, its two per-pair meshes or (unless *skip_nifti*)
        the subject's T1 are missing.
    FileExistsError
        If outputs exist and *overwrite* is false.
    ValueError
        If the simulation is not two-pair TI, is carrier-only, or its two
        meshes do not pair element by element.
    """
    import tit
    from simnibs import mesh_io

    from tit.paths import get_path_manager

    pm = get_path_manager(project)
    m2m_dir = pm.m2m(subject)
    sim_dir = pm.simulation(subject, sim)
    config_path = os.path.join(sim_dir, "documentation", "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"No simulation {sim!r} for subject {subject!r}: {config_path} not found."
        )
    with open(config_path) as fh:
        cfg = json.load(fh)
    check_simulation_config(cfg)

    cond = cfg.get("conductivity", "scalar")
    hf = [
        os.path.join(
            sim_dir, "high_Frequency", "mesh", f"{subject}_TDCS_{i}_{cond}.msh"
        )
        for i in (1, 2)
    ]
    missing = [p for p in hf if not os.path.isfile(p)]
    if not skip_nifti and not os.path.isfile(pm.t1(subject)):
        missing.append(pm.t1(subject))
    if missing:
        raise FileNotFoundError(
            "Missing inputs (the per-pair meshes are written by the simulator "
            "under high_Frequency/mesh/):\n  " + "\n  ".join(missing)
        )
    nifti_dir = os.path.join(sim_dir, "TI", "niftis")
    targets = output_paths(nifti_dir, sim, skip_nifti)
    existing = [p for p in targets if os.path.exists(p)]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs exist (pass --overwrite to replace them):\n  "
            + "\n  ".join(existing)
        )

    print(f"Reading {hf[0]}\n        {hf[1]}", flush=True)
    keep = [const.WM_TISSUE_TAG, const.GM_TISSUE_TAG]
    m1 = mesh_io.read_msh(hf[0]).crop_mesh(tags=keep)
    m2 = mesh_io.read_msh(hf[1]).crop_mesh(tags=keep)
    pairing = {
        "same_element_count": m1.elm.nr == m2.elm.nr,
        "same_connectivity": bool(
            np.array_equal(m1.elm.node_number_list, m2.elm.node_number_list)
        ),
        "same_nodes": bool(np.allclose(m1.nodes.node_coord, m2.nodes.node_coord)),
    }
    if not all(pairing.values()):
        raise ValueError(f"Cropped meshes do not pair element by element: {pairing}")
    E1, E2 = m1.field["E"].value, m2.field["E"].value
    print(f"{m1.elm.nr} GM+WM elements; computing metrics", flush=True)
    met = am_phase_metrics(E1, E2)
    vols = m1.elements_volumes_and_areas().value
    print("Brute-force check", flush=True)
    bf = brute_force_check(E1, E2, met, n=n_check)
    print(json.dumps(bf, indent=2), flush=True)

    report: dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "tool": f"tit.tools.am_phase (TI-Toolbox {tit.__version__})",
        "subject": subject,
        "simulation": sim,
        "inputs": hf,
        "electrode_pairs": cfg["electrode_pairs"],
        "intensities": cfg.get("intensities"),
        "polarity": "first electrode of each pair is +",
        "definitions": {
            **DEFINITIONS,
            "lut_colours": {
                "1": "#%02x%02x%02x" % _COL_PSI0,
                "2": "#%02x%02x%02x" % _COL_PSIPI,
            },
        },
        "mesh_pairing": pairing,
        "element_volumes": {
            "volume_cm3_psi0": float(vols[met["label"] == LABEL_PSI0].sum() / 1000),
            "volume_cm3_psipi": float(vols[met["label"] == LABEL_PSIPI].sum() / 1000),
        },
        "brute_force_check": bf,
    }

    if not skip_nifti:
        print("Writing NIfTIs", flush=True)
        paths, checks = write_niftis(
            copy.deepcopy(m1),
            met,
            m2m_dir,
            pm.t1(subject),
            nifti_dir,
            sim,
            labeling_path=pm.tissue_labeling(subject),
            roi_label=roi_label,
        )
        report["niftis"] = paths
        report["nifti_checks"] = checks
        print(json.dumps(checks, indent=2), flush=True)

    with open(targets[0], "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {targets[0]}")
    report["report_path"] = targets[0]
    return report


def _report_job_outputs(report: dict, json_path: str) -> None:
    """Announce the written files to the job manager, if running inside a job.

    A no-op outside one, and never fatal: the files are already on disk.
    """
    try:
        from tit.jobs import events

        for key, path in report.get("niftis", {}).items():
            kind = "txt" if path.endswith(".txt") else "nifti"
            events.emit_artifact(path, kind=kind, label=f"AM phase {key}")
        events.emit_artifact(json_path, kind="json", label="AM phase report")
        events.emit_result({"success": True, "report": json_path})
    except Exception as exc:  # noqa: BLE001 - the files are written; reporting is extra
        print(f"artifact reporting skipped: {exc}")


def main(argv: list[str] | None = None) -> None:
    """``simnibs_python -m tit.tools.am_phase --subject <id> --sim <name> [options]``."""
    ap = argparse.ArgumentParser(
        prog="simnibs_python -m tit.tools.am_phase",
        description=(
            "AM-phase maps for a finished two-pair TI simulation: where in each "
            "beat the field-magnitude envelope peaks (label), the envelope depth "
            "and the angle between the two fields, written as NIfTIs beside the "
            "simulation's own. Run inside the TI-Toolbox container."
        ),
        epilog=(
            "example: simnibs_python -m tit.tools.am_phase --subject 101 "
            "--sim L_Insula --roi-label 53\n"
            "outputs: <simulation>/TI/niftis/<sim>_AMphase_*. "
            "Documentation: the Simulator wiki page, section 'AM phase'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--subject", required=True, help="subject ID, without 'sub-'")
    ap.add_argument("--sim", required=True, help="simulation name")
    ap.add_argument(
        "--project",
        default=None,
        help="project root (default: auto-detected inside the container)",
    )
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    ap.add_argument("--skip-nifti", action="store_true", help="write only the JSON")
    ap.add_argument(
        "--n-check",
        type=int,
        default=2000,
        help="elements sampled by the brute-force check (default: 2000)",
    )
    ap.add_argument(
        "--roi-label",
        type=int,
        default=None,
        help="value in m2m_<subject>/segmentation/labeling.nii.gz to summarise",
    )
    args = ap.parse_args(argv)
    try:
        report = run_am_phase(
            args.subject,
            args.sim,
            project=args.project,
            overwrite=args.overwrite,
            skip_nifti=args.skip_nifti,
            n_check=args.n_check,
            roi_label=args.roi_label,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        sys.exit(str(exc))
    _report_job_outputs(report, report["report_path"])


if __name__ == "__main__":
    main()
