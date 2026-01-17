# semantically-guided-video-compression
## Overview
Video compression algorithm which seeks to compress video by finding the subset of frames/clips which retain the semantics of the video.  Assumes posed rgb frames as an input.  Applications vary, but efficient memory for robots over long time horizons (hours, days, months) is the motivation. 

This work focuses on evaluating lightweight signals that can be used for online frame selection.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c63b50eb-abf7-4536-a2d1-66552acc590d" />
</p>

**Links**

- [paper](https://antony-blog.notion.site/Semantic-Retention-under-Compression-Efficient-Memory-for-Mobile-Robots-2e8a5377d6f18079bcbfeb78d857ff93?source=copy_link)
- [robot trajectory video](https://antony-blog.notion.site/TSRB-Mission-1-Video-2c5a5377d6f18020b3a3d377f12b67a0?source=copy_link)

## Research Positioning 
I consciously position this paper as exploratory, rather than ”novel method best thing ever”. I think the biggest contributions my paper makes are

- Introducing the idea of compact robot memory as a problem in its own right. Video compression is well-studied, but thinking about it from a robot-memory point of view where every frame is posed and tied to the robot’s trajectory is new and I think useful.
- Showing that pose and embeddings are the two most salient signals for compressing robot memory in practice. They’re lightweight, easy to compute, and together cover most of what actually matters for keeping the “story” of a deployment.
• Putting forward a simple evaluation metric for semantic retention in compressed video. It’s not perfect, but it gives a starting point, and I hope others build on it and make something more task-aligned and robust.

## Compression Methods
I investigate 4 compression methods:
1. **Naive Temporal Downsampling**
2. **Pose Based:** Enforce no viewpoint redudancy (with some resolution)
3. **Semantic Based**: Enforce no repeated visual modes.  Visual modes represented by lightweight Dino embedding.
4. **Pose + Semantic Hybrid**: For a given viewpoint, only add frame to memory if it is sufficiently semantically distinct from other frames at this viewpoint

## Evaluation Methods
Compression techniques are evaluated along 4 axes:
1. **Compression Ratio:** (# of frames in compressed set / # of frames in original set)
2. **Geometric Coverage:** Discretize covered space into x,y,yaw bins, metric is (# of pose cells covered in compressed set / # of pose cells covered in original set)
3. **Semantic Coverage:** Compute heavy weight embeddings per frame (Clip-G to prevent circularity w/ Dino selection), k-means cluster.  These clusters represent distinct visual modes. metric is: (# of clusters in covered in compressed set / # of total clusters)
4. **Rare Cluster Recall**: Using the embeddings from above, rare clusters 25% smallest clusters (least frequenct visual modes).  Metric is (# of rare clusters retained in compressed set / # of rare clusters)


## TL;DR

Geometric and semantic cues are the way to go. They’re surprisingly good at keeping visual diversity high, especially in places with a lot of repetitive structures (like hallways or warehouses).

Temporal sampling is helpful but should be used with caution. It works fine if the robot is mostly sitting still or on short runs, but if downsampled too aggresively will cut out information from parts of the trajectory where the robot moved quickly, or dynamic events occur.

You don't need massive overhead to get decent compression. Simple geometric and semantic methods can shrink video data significantly while still keeping all the "meaning" intact for downstream tasks.


## Learnings
For me the hardest part of this entire process was finding a research problem that was unexplored, but was sufficiently simple that I thought I could contribute in the course of a semester. I spent roughly 55% of my time reading papers in the area and finding direction, 30% of my time writing infrastucture code (well written tools for iterating through rosbags, ros to python object conversions, creating datasets, visualizations, evaluations etc), and the remaining 15% of my time actually writing the algorithms for this paper. In a sense I felt rushed, but I’m also happy I took the time to find a direction others haven’t explored (that I’ve found). 

One important learning is on evaluation metrics. For this paper I used CLIP embedding based clustering to evaluate the retention of visual modes in compressed datasets. While this metric is valid, it doesn’t measure the true thing we’d like to measure: the semantic content of the extracted captions from the VLMs that will run on the compressed datasets, which is what actually ends up being used in robot memories. With more time I would have written an evaluation metric that ran compressed datasets through a captioner and evaluated semantic retention of the compressed caption set vs. the full caption set, but this would have required more work than I had time for. 

This brings me to my next learning: keeping myself in a reasonable scope was tough. I wanted to write a better evaluation metric, I wanted to come up with a more interesting/ complex method for compression, and I wanted to record and analyze more data. At the end of the day, I was forced to abandon some of these pursuits as I only had so much time, and at a certain point I had to stop optimizing for quality of work, and start optimizing for finishing. Going forwards I’d like to do a better job of keeping myself in scope.
