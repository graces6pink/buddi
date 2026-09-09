# FlickrCI3D contact-only comparison

- images considered: 150
- images with a result from every run (used below): 133
- binary contact threshold: 0.013 m

## Coverage

| run | results | no result | bbox mismatch |
| --- | --- | --- | --- |
| baseline-nodiff | 135 | 12 | 3 |
| buddi-official | 133 | 12 | 5 |
| buddi-mine | 135 | 12 | 3 |
| buddi-ep323 | 135 | 12 | 3 |

## Metrics on the common subset

| run | iou | fscore | precision | recall | dist_on_gt_mm | penetration_mm | pcc@0.10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-nodiff | 0.0039 | 0.0066 | 0.0128 | 0.0063 | 255.5437 | 22.6425 | 0.2681 |
| buddi-official | 0.0062 | 0.0092 | 0.0188 | 0.0064 | 182.7576 | 34.3246 | 0.3705 |
| buddi-mine | 0.0006 | 0.0011 | 0.0015 | 0.0008 | 249.0807 | 33.1696 | 0.2617 |
| buddi-ep323 | 0.0043 | 0.0069 | 0.0188 | 0.0044 | 245.8934 | 34.3350 | 0.2866 |
