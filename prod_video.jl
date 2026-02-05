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

    image_values = [float(second_raster[j, i].val) for i in axes(second_raster, 2), j in axes(second_raster, 1)]

    x_vec = 1:size(image_values, 1)
    y_vec = 1:size(image_values, 2)

    image_values .= reverse(image_values, dims=2)
    (xs, ys) = flock_outline(x_vec, y_vec, image_values) # coordinates of this line segment

    for (x, y) in zip(floor.(Int, xs), floor.(Int, ys))
        y_max = size(raw_img, 1)
        raw_img[y_max - y-1, x] = colorant"firebrick2" # Red color for the outline
        raw_img[y_max - y, x] = colorant"firebrick2" # Red color for the outline
        raw_img[y_max - y+1, x] = colorant"firebrick2" # Red color for the outline
    end


    # Draw the raster
    if 25 < i < 450
        # Get the midpoint
        midpoint = floor.(Int, flock_midpoint(xs, ys)) .+ (-70, -40)

        # Insert dimensions
        insert_x = 100
        insert_y = 150

        # Insert indices
        insert_x_indices = clamp.(range(midpoint[1] - insert_x, midpoint[1] + insert_x), 1, size(raw_img, 2))
        insert_y_indices = clamp.(range(midpoint[2] - insert_y, midpoint[2] + insert_y), 1, size(raw_img, 1))

        # Draw the insert 
        for x in insert_x_indices, y in insert_y_indices
            raw_img[y, x] = second_raster[y, x]
        end
    end


    if 480 < i < 601
        # Get the midpoint
        midpoint = floor.(Int, flock_midpoint(xs, ys)) .+ (130, -220)

        # Insert dimensions
        insert_x = 100
        insert_y = 150

        # Insert indices
        insert_x_indices = clamp.(range(midpoint[1] - insert_x, midpoint[1] + insert_x), 1, size(raw_img, 2))
        insert_y_indices = clamp.(range(midpoint[2] - insert_y, midpoint[2] + insert_y), 1, size(raw_img, 1))

        # Draw the insert 
        for x in insert_x_indices, y in insert_y_indices
            raw_img[y, x] = second_raster[y, x]
        end
    end



    if  350 < i < 570
        # Get the midpoint
        midpoint = floor.(Int, flock_midpoint(xs, ys)) .+ (-50, 220)

        # Insert dimensions
        insert_x = 240
        insert_y = 60

        # Insert indices
        insert_x_indices = clamp.(range(midpoint[1] - insert_x, midpoint[1] + insert_x), 1, size(raw_img, 2))
        insert_y_indices = clamp.(range(midpoint[2] - insert_y, midpoint[2] + insert_y), 1, size(raw_img, 1))

        # Draw the insert 
        for x in insert_x_indices, y in insert_y_indices
            if second_raster[y, x].val > 0.2
                raw_img[y, x] = colorant"black" # Blue color for the raster
            else
                raw_img[y, x] = colorant"slateblue1" # Black color for the background
            end
        end


    end

    push!(final_film, copy(raw_img))

end

# Save film
encoder_options = (crf=23, preset="medium")
VideoIO.save("flock_film.mp4", final_film, framerate=30, encoder_options=encoder_options)
