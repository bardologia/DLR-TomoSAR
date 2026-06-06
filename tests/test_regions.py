from __future__ import annotations

import pytest

from tools.regions import CropRegion, SplitRegions


class TestCropRegionConstruction:
    def test_valid_region_constructs(self):
        region = CropRegion(0, 10, 0, 20)

        assert region.azimuth_start == 0
        assert region.azimuth_end   == 10
        assert region.range_start   == 0
        assert region.range_end     == 20

    def test_azimuth_start_equal_to_end_raises(self):
        with pytest.raises(ValueError, match="azimuth_start"):
            CropRegion(5, 5, 0, 10)

    def test_azimuth_start_greater_than_end_raises(self):
        with pytest.raises(ValueError, match="azimuth_start"):
            CropRegion(10, 5, 0, 10)

    def test_range_start_equal_to_end_raises(self):
        with pytest.raises(ValueError, match="range_start"):
            CropRegion(0, 10, 7, 7)

    def test_range_start_greater_than_end_raises(self):
        with pytest.raises(ValueError, match="range_start"):
            CropRegion(0, 10, 12, 4)

    def test_minimal_unit_region_constructs(self):
        region = CropRegion(0, 1, 0, 1)

        assert region.azimuth_size == 1
        assert region.range_size   == 1

    def test_negative_coordinates_allowed_when_ordered(self):
        region = CropRegion(-10, -5, -20, -3)

        assert region.azimuth_size == 5
        assert region.range_size   == 17


class TestCropRegionAccessors:
    def test_as_tuple_order(self):
        region = CropRegion(1, 2, 3, 4)

        assert region.as_tuple() == (1, 2, 3, 4)

    def test_as_identifier_string_joins_with_a(self):
        region = CropRegion(1, 2, 3, 4)

        assert region.as_identifier_string() == "1a2a3a4"

    def test_as_identifier_string_with_negative_values(self):
        region = CropRegion(-1, 2, -3, 4)

        assert region.as_identifier_string() == "-1a2a-3a4"

    def test_as_labeled_string_separates_axes(self):
        region = CropRegion(1000, 16000, 500, 4000)

        assert region.as_labeled_string() == "az1000-16000_rg500-4000"

    def test_azimuth_size(self):
        region = CropRegion(5, 17, 0, 1)

        assert region.azimuth_size == 12

    def test_range_size(self):
        region = CropRegion(0, 1, 8, 30)

        assert region.range_size == 22


class TestCropRegionLocalSlices:
    def test_local_slices_relative_to_global(self):
        local      = CropRegion(10, 20, 30, 40)
        global_box = CropRegion(5, 50, 25, 60)

        az_slice, rg_slice = local.local_slices(global_box)

        assert az_slice == slice(5, 15)
        assert rg_slice == slice(5, 15)

    def test_local_slices_identical_global_starts_at_zero(self):
        region = CropRegion(10, 20, 30, 40)

        az_slice, rg_slice = region.local_slices(region)

        assert az_slice == slice(0, 10)
        assert rg_slice == slice(0, 10)

    def test_local_slices_lengths_match_sizes(self):
        local      = CropRegion(12, 18, 22, 35)
        global_box = CropRegion(0, 100, 0, 100)

        az_slice, rg_slice = local.local_slices(global_box)

        assert az_slice.stop - az_slice.start == local.azimuth_size
        assert rg_slice.stop - rg_slice.start == local.range_size


class TestCropRegionSubdivideByAzimuth:
    def test_subdivide_exact_multiple(self):
        region = CropRegion(0, 20, 0, 5)

        subs = region.subdivide_by_azimuth(10)

        assert len(subs) == 2
        assert subs[0].as_tuple() == (0, 10, 0, 5)
        assert subs[1].as_tuple() == (10, 20, 0, 5)

    def test_subdivide_with_remainder(self):
        region = CropRegion(0, 25, 0, 5)

        subs = region.subdivide_by_azimuth(10)

        assert len(subs) == 3
        assert subs[-1].as_tuple() == (20, 25, 0, 5)

    def test_subdivide_max_width_larger_than_region(self):
        region = CropRegion(0, 8, 0, 5)

        subs = region.subdivide_by_azimuth(100)

        assert len(subs) == 1
        assert subs[0].as_tuple() == region.as_tuple()

    def test_subdivide_max_width_one(self):
        region = CropRegion(0, 4, 0, 5)

        subs = region.subdivide_by_azimuth(1)

        assert len(subs) == 4
        assert [s.azimuth_start for s in subs] == [0, 1, 2, 3]
        assert [s.azimuth_end   for s in subs] == [1, 2, 3, 4]

    def test_subdivide_preserves_range_bounds(self):
        region = CropRegion(0, 30, 7, 19)

        subs = region.subdivide_by_azimuth(8)

        for sub in subs:
            assert sub.range_start == 7
            assert sub.range_end   == 19

    def test_subdivide_is_contiguous_and_covers_region(self):
        region = CropRegion(3, 27, 0, 5)

        subs = region.subdivide_by_azimuth(7)

        assert subs[0].azimuth_start == region.azimuth_start
        assert subs[-1].azimuth_end  == region.azimuth_end
        for previous, following in zip(subs, subs[1:]):
            assert previous.azimuth_end == following.azimuth_start

    def test_subdivide_offset_start(self):
        region = CropRegion(100, 130, 0, 5)

        subs = region.subdivide_by_azimuth(10)

        assert len(subs) == 3
        assert subs[0].azimuth_start == 100
        assert subs[-1].azimuth_end  == 130


class TestSplitRegionsAsList:
    def test_as_list_wraps_single_region(self):
        region = CropRegion(0, 10, 0, 10)

        result = SplitRegions.as_list(region)

        assert result == [region]

    def test_as_list_passes_through_list(self):
        regions = [CropRegion(0, 10, 0, 10), CropRegion(10, 20, 0, 10)]

        result = SplitRegions.as_list(regions)

        assert result == regions
        assert result is not regions

    def test_as_list_converts_tuple_to_list(self):
        regions = (CropRegion(0, 10, 0, 10), CropRegion(10, 20, 0, 10))

        result = SplitRegions.as_list(regions)

        assert result == list(regions)
        assert isinstance(result, list)


class TestSplitRegionsItemsAndRegions:
    def test_items_returns_named_pairs(self):
        train = CropRegion(0, 10, 0, 10)
        val   = CropRegion(10, 15, 0, 10)
        test  = CropRegion(15, 20, 0, 10)
        split = SplitRegions(train, val, test)

        assert split.items() == [("train", train), ("val", val), ("test", test)]

    def test_region_lists_wraps_each_split(self):
        train = CropRegion(0, 10, 0, 10)
        val   = [CropRegion(10, 15, 0, 10), CropRegion(15, 18, 0, 10)]
        test  = CropRegion(18, 20, 0, 10)
        split = SplitRegions(train, val, test)

        named_lists = split.region_lists()

        assert named_lists[0] == ("train", [train])
        assert named_lists[1] == ("val", val)
        assert named_lists[2] == ("test", [test])

    def test_regions_lookup_by_name(self):
        train = CropRegion(0, 10, 0, 10)
        val   = CropRegion(10, 15, 0, 10)
        test  = CropRegion(15, 20, 0, 10)
        split = SplitRegions(train, val, test)

        assert split.regions("train") == [train]
        assert split.regions("val")   == [val]
        assert split.regions("test")  == [test]

    def test_regions_lookup_returns_list_for_list_split(self):
        train = [CropRegion(0, 5, 0, 10), CropRegion(5, 10, 0, 10)]
        val   = CropRegion(10, 15, 0, 10)
        test  = CropRegion(15, 20, 0, 10)
        split = SplitRegions(train, val, test)

        assert split.regions("train") == train

    def test_regions_unknown_name_raises_key_error(self):
        split = SplitRegions(
            CropRegion(0, 10, 0, 10),
            CropRegion(10, 15, 0, 10),
            CropRegion(15, 20, 0, 10),
        )

        with pytest.raises(KeyError):
            split.regions("unknown")


class TestSplitRegionsBoundingGlobalCrop:
    def test_bounding_crop_spans_all_regions(self):
        split = SplitRegions(
            train = CropRegion(0, 10, 5, 20),
            val   = CropRegion(10, 15, 0, 12),
            test  = CropRegion(15, 25, 8, 30),
        )

        bounding = split.bounding_global_crop()

        assert bounding.azimuth_start == 0
        assert bounding.azimuth_end   == 25
        assert bounding.range_start   == 0
        assert bounding.range_end     == 30

    def test_bounding_crop_with_list_splits(self):
        split = SplitRegions(
            train = [CropRegion(0, 5, 0, 10), CropRegion(30, 40, 0, 10)],
            val   = CropRegion(10, 15, 2, 18),
            test  = CropRegion(15, 20, 0, 25),
        )

        bounding = split.bounding_global_crop()

        assert bounding.azimuth_start == 0
        assert bounding.azimuth_end   == 40
        assert bounding.range_start   == 0
        assert bounding.range_end     == 25

    def test_bounding_crop_single_region_returns_equivalent_bounds(self):
        region = CropRegion(3, 14, 6, 22)
        split  = SplitRegions(region, region, region)

        bounding = split.bounding_global_crop()

        assert bounding.as_tuple() == region.as_tuple()


class TestSplitRegionsFromRatios:
    def test_from_ratios_default_split_sizes(self):
        global_crop = CropRegion(0, 100, 0, 50)

        split = SplitRegions.from_ratios(global_crop)

        assert split.train.as_tuple() == (0, 70, 0, 50)
        assert split.val.as_tuple()   == (70, 85, 0, 50)
        assert split.test.as_tuple()  == (85, 100, 0, 50)

    def test_from_ratios_covers_full_azimuth_range(self):
        global_crop = CropRegion(0, 100, 0, 50)

        split = SplitRegions.from_ratios(global_crop)

        assert split.train.azimuth_start == global_crop.azimuth_start
        assert split.test.azimuth_end    == global_crop.azimuth_end
        assert split.train.azimuth_end   == split.val.azimuth_start
        assert split.val.azimuth_end     == split.test.azimuth_start

    def test_from_ratios_preserves_range_bounds(self):
        global_crop = CropRegion(0, 100, 7, 41)

        split = SplitRegions.from_ratios(global_crop)

        for region in (split.train, split.val, split.test):
            assert region.range_start == 7
            assert region.range_end   == 41

    def test_from_ratios_with_offset_start(self):
        global_crop = CropRegion(200, 300, 0, 10)

        split = SplitRegions.from_ratios(global_crop)

        assert split.train.azimuth_start == 200
        assert split.train.azimuth_end   == 200 + int(100 * 0.70)
        assert split.test.azimuth_end    == 300

    def test_from_ratios_custom_ratios(self):
        global_crop = CropRegion(0, 200, 0, 10)

        split = SplitRegions.from_ratios(global_crop, train_ratio=0.5, val_ratio=0.25)

        assert split.train.as_tuple() == (0, 100, 0, 10)
        assert split.val.as_tuple()   == (100, 150, 0, 10)
        assert split.test.as_tuple()  == (150, 200, 0, 10)

    def test_from_ratios_bounding_recovers_global_crop(self):
        global_crop = CropRegion(0, 100, 0, 50)

        split    = SplitRegions.from_ratios(global_crop)
        bounding = split.bounding_global_crop()

        assert bounding.as_tuple() == global_crop.as_tuple()

    def test_from_ratios_zero_train_ratio_raises_on_empty_train(self):
        global_crop = CropRegion(0, 100, 0, 10)

        with pytest.raises(ValueError, match="azimuth_start"):
            SplitRegions.from_ratios(global_crop, train_ratio=0.0, val_ratio=0.5)
