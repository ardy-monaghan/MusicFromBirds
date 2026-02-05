module FlockingAnalysis
using Contour, Statistics

export flock_outline, flock_deviations

function flock_outline(x_vec, y_vec, image_values)
    median_contour = Contour.levels(contours(x_vec,y_vec,image_values, 1))[1]
    line_vec = Contour.lines(median_contour)

    return coordinates(argmax(t -> length(t.vertices), line_vec))
end

function flock_midpoint(xs, ys)
    return ((minimum(xs) + maximum(xs)) / 2, (minimum(ys) + maximum(ys)) / 2)
end

function distance2centre(xs, ys)
    mx, my = flock_midpoint(xs, ys)
    dists = sqrt.((xs .- mx).^2 .+ (ys .- my).^2)
    return dists
end

function flock_deviations(xs, ys)
    dists = distance2centre(xs, ys)
    mean_dist = mean(dists)
    deviations = dists .- mean_dist
    return deviations
end

end # module FlockingAnalysis
