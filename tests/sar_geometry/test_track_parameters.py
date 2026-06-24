from __future__ import annotations

from pathlib import Path

import pytest

from tools.sar.track_parameters import (
    StepParameterFile,
    StepParameterResolver,
    TrackParameterCollector,
    TrackParameters,
)


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<idl2xml>
  <object name="step_processing_parameters">
    <parameter name="ident">
      <datatype length="1">string</datatype>
      <value>SARTOM0102</value>
    </parameter>
    <parameter name="antdir">
      <datatype length="1">long</datatype>
      <value>1</value>
    </parameter>
    <parameter name="lambda">
      <datatype length="1">double</datatype>
      <value>2.26195004272000011e-1</value>
    </parameter>
    <parameter name="da">
      <datatype length="1">double</datatype>
      <value>6.10865238197999982e-1</value>
    </parameter>
    <parameter name="h0">
      <datatype length="1">double</datatype>
      <value>3.71915192910869674e3</value>
    </parameter>
    <parameter name="terrain">
      <datatype length="1">double</datatype>
      <value>6.83882507324218736e2</value>
    </parameter>
    <parameter name="rref">
      <datatype length="1">double</datatype>
      <value>4.58523851573264007e3</value>
    </parameter>
    <parameter name="ang_range">
      <datatype length="2">double</datatype>
      <value>[1.50000000000000000e1,9.00000000000000000e1]</value>
    </parameter>
    <parameter name="r">
      <datatype length="1">pointer</datatype>
      <value>
        <parameter name="ptr">
          <datatype length="3">double</datatype>
          <value>[3.30000000000000000e3,4.00000000000000000e3,6.00000000000000000e3]</value>
        </parameter></value>
    </parameter>
    <parameter name="dims_info">
      <datatype length="1">struct</datatype>
      <value>
        <object name="step_dims_info">
          <parameter name="side_looking">
            <datatype length="1">string</datatype>
            <value>right</value>
          </parameter>
        </object></value>
    </parameter>
  </object>
</idl2xml>
"""


def _write_sample(directory, pols=("hh",), primary=True) -> None:
    subdir  = Path("GTC") / "GTC-RDP" if primary else Path("INF") / "INF-RDP"
    product = directory / subdir
    product.mkdir(parents=True)

    for pol in pols:
        (product / f"pp_17sartom0102_L{pol}_t01L.xml").write_text(SAMPLE_XML, encoding="utf-8")


def test_parser_coerces_scalars_arrays_and_nested(tmp_path):
    path = tmp_path / "pp.xml"
    path.write_text(SAMPLE_XML, encoding="utf-8")

    params = StepParameterFile(path).parse()

    assert params["ident"]                    == "SARTOM0102"
    assert params["antdir"]                   == 1
    assert params["lambda"]                   == pytest.approx(0.226195, rel=1e-5)
    assert params["ang_range"]                == [15.0, 90.0]
    assert params["r"]                        == [3300.0, 4000.0, 6000.0]
    assert params["dims_info"]["side_looking"] == "right"


def test_resolver_finds_primary_under_gtc_rdp(tmp_path):
    _write_sample(tmp_path / "PS02" / "T01L", primary=True)

    resolved = StepParameterResolver().resolve_for_polarisation(tmp_path / "PS02" / "T01L", "hh", is_primary=True)

    assert resolved.parent.parts[-2:] == ("GTC", "GTC-RDP")
    assert resolved.name              == "pp_17sartom0102_Lhh_t01L.xml"


def test_resolver_finds_secondary_under_inf_rdp(tmp_path):
    _write_sample(tmp_path / "PS04" / "T01L", primary=False)

    resolved = StepParameterResolver().resolve_for_polarisation(tmp_path / "PS04" / "T01L", "hh", is_primary=False)

    assert resolved.parent.parts[-2:] == ("INF", "INF-RDP")
    assert resolved.name              == "pp_17sartom0102_Lhh_t01L.xml"


def test_resolver_does_not_cross_roles(tmp_path):
    _write_sample(tmp_path / "PS02" / "T01L", primary=True)

    with pytest.raises(FileNotFoundError):
        StepParameterResolver().resolve_for_polarisation(tmp_path / "PS02" / "T01L", "hh", is_primary=False)


def test_resolver_selects_requested_polarisation(tmp_path):
    _write_sample(tmp_path / "PS02" / "T01L", pols=("hh", "hv", "vv"), primary=True)

    resolved = StepParameterResolver().resolve_for_polarisation(tmp_path / "PS02" / "T01L", "hv", is_primary=True)

    assert resolved.name == "pp_17sartom0102_Lhv_t01L.xml"


def test_resolver_raises_when_polarisation_absent(tmp_path):
    _write_sample(tmp_path / "PS02" / "T01L", pols=("hh", "vv"), primary=True)

    with pytest.raises(FileNotFoundError):
        StepParameterResolver().resolve_for_polarisation(tmp_path / "PS02" / "T01L", "hv", is_primary=True)


def test_collector_builds_parameters_over_passes(tmp_path):
    _write_sample(tmp_path / "FL01" / "PS02" / "T01L", pols=("hh", "hv"), primary=True)
    _write_sample(tmp_path / "FL01" / "PS04" / "T01L", pols=("hh", "hv"), primary=False)

    directories = [tmp_path / "FL01" / "PS02" / "T01L", tmp_path / "FL01" / "PS04" / "T01L"]
    parameters  = TrackParameterCollector.from_pass_directories(directories, "hv").collect()

    assert parameters.labels      == ["FL01_PS02", "FL01_PS04"]
    assert parameters.reference   == "FL01_PS02"
    assert parameters.n_tracks    == 2
    assert parameters.track_files[0].endswith("GTC/GTC-RDP/pp_17sartom0102_Lhv_t01L.xml")
    assert parameters.track_files[1].endswith("INF/INF-RDP/pp_17sartom0102_Lhv_t01L.xml")


def test_derived_geometry_matches_acquisition(tmp_path):
    path = tmp_path / "pp.xml"
    path.write_text(SAMPLE_XML, encoding="utf-8")

    parameters = TrackParameters(labels=["FL01_PS02"], parameters=[StepParameterFile(path).parse()])
    geometry   = parameters.derived()[0]

    assert geometry["look_side"]            == "right"
    assert geometry["depression_angle_deg"] == pytest.approx(35.0, abs=1e-6)
    assert geometry["slant_range_near_m"]   == 3300.0
    assert geometry["slant_range_far_m"]    == 6000.0
    assert geometry["look_angle_far_deg"]   > geometry["look_angle_near_deg"]


def test_payload_roundtrip_preserves_parameters(tmp_path):
    path = tmp_path / "pp.xml"
    path.write_text(SAMPLE_XML, encoding="utf-8")

    parameters = TrackParameters(labels=["FL01_PS02"], parameters=[StepParameterFile(path).parse()])
    saved      = parameters.save(tmp_path / TrackParameters.FILENAME)
    restored   = TrackParameters.load(saved)

    assert restored.labels                       == parameters.labels
    assert restored.parameters[0]["r"]           == [3300.0, 4000.0, 6000.0]
    assert restored.parameters[0]["dims_info"]   == {"side_looking": "right"}
