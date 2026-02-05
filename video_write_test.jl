using FlockingAnalysis, VideoIO, ImageBinarization, Contour, CairoMakie, ImageFiltering, Statistics, Colors

starlings = VideoIO.load("starlings.mp4")

rough_alg = AdaptiveThreshold(window_size = 4; percentage = 1)
fine_alg = AdaptiveThreshold(window_size = 250; percentage = 0.01)

chopped_starlings = starlings[1:600]

final_film = Matrix{typeof(chopped_starlings[1][1])}[]

# Loop over every frame
for (i, raw_img) in enumerate(chopped_starlings)  

    first_blur = imfilter(raw_img, Kernel.gaussian(1))
    img = binarize(first_blur, rough_alg)
    second_blur = imfilter(img, Kernel.gaussian(12))
    second_raster = binarize(second_blur, fine_alg)

    image_values = [float(second_raster[j, i].val) for i in 1:size(second_raster, 2), j in 1:size(second_raster, 1)]

    x_vec = 1:size(image_values, 1)
    y_vec = 1:size(image_values, 2)

    image_values .= reverse(image_values, dims=2)
    (xs, ys) = flock_outline(x_vec, y_vec, image_values) # coordinates of this line segment

    # Colour the outline
    for (x, y) in zip(floor.(Int, xs), floor.(Int, ys))
        y_max = size(raw_img, 1)
        raw_img[y_max - y-1, x] = colorant"firebrick2" # Red color for the outline
        raw_img[y_max - y, x] = colorant"firebrick2" # Red color for the outline
        raw_img[y_max - y+1, x] = colorant"firebrick2" # Red color for the outline
    end

    push!(final_film, copy(raw_img))
end

## Save film?

encoder_options = (crf=23, preset="medium")
VideoIO.save("video.mp4", final_film, framerate=30, encoder_options=encoder_options)



