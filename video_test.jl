using VideoIO, ImageBinarization, Contour, CairoMakie, ImageFiltering, Statistics

starlings = VideoIO.load("starlings.mp4")

rough_alg = AdaptiveThreshold(window_size = 4; percentage = 1)
fine_alg = AdaptiveThreshold(window_size = 250; percentage = 0.01)

raw_img = starlings[350]
first_blur = imfilter(raw_img, Kernel.gaussian(1))
img = binarize(first_blur, rough_alg)
second_blur = imfilter(img, Kernel.gaussian(12))
second_raster = binarize(second_blur, fine_alg)


image_values = [float(second_raster[j, i].val) for i in 1:size(second_raster, 2), j in 1:size(second_raster, 1)]

x_vec = 1:size(image_values, 1)
y_vec = 1:size(image_values, 2)

# c = contour(x_vec, y_vec, image_values, 0.5)

image_values .= reverse(image_values, dims=2)
(xs, ys) = flock_outline(x_vec, y_vec, image_values) # coordinates of this line segment

fig = Figure(size = (length(x_vec),length(y_vec)))
ax = Axis(fig[1, 1])

plotting_image = reverse(raw_img, dims=1)
image!(ax, plotting_image')

# CairoMakie.heatmap!(ax, x_vec, y_vec, image_values)

lines!(ax, xs, ys, color = :red, linewidth = 1)

display(fig)
