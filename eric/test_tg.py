import tgt

grid = tgt.TextGrid(filename="test")
tier=tgt.IntervalTier(start_time=0,end_time=5,name="mot")
label=tgt.core.Interval(2, 3,"word")
tier.add_annotation(label)
grid.add_tier(tier)
tgt.write_to_file(grid, "/home/leferrae/Desktop/These/test.textgrid")