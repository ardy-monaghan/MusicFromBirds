using VideoIO, ImageBinarization, Contour, CairoMakie, ImageFiltering, Statistics, PortAudio, Interpolations

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

function phase_matching(signal, reference_signal, threshold)


    function error_at_offset(signal, reference_signal, offset, idx)
        shifted_signal = vcat(signal[idx:end], signal[1:idx-1])

        return sum(abs.(shifted_signal[1:offset] .- reference_signal[1:offset]))
    end


    # Find the minimum error offset
    lag = findmin( i -> error_at_offset(signal, reference_signal, threshold, i),  1:(length(signal)-threshold))[2]

    aligned_signal = vcat(signal[lag:end], signal[1:lag-1])

    aligned_signal .+= (reference_signal[1] .- aligned_signal[1])    

    return aligned_signal
end

function fill_bald_spots!(signal, padding)

    out_signal = Float64[]
    jump_idx = findall(s -> abs(s) > 0.01, diff(signal))

    # Fully circular signal, no jumps
    if abs(signal[end] - signal[1]) < 0.01 && isempty(jump_idx)
        return signal

    # Signal is not circular
    elseif isempty(jump_idx)

        # Cut the data in half
        cut_location = floor(Int, length(signal)/2)
        cut_data = vcat(signal[cut_location+1:end], signal[1:cut_location])

        # Cubic interpolation
        x = 1:length(cut_data)
        itp_cubic = cubic_spline_interpolation(x, cut_data)

        # Fake an increased density in between the points of interest
        x_new = vcat(1:cut_location, range(cut_location, cut_location+1, padding)[2:end-1], (cut_location+1):length(cut_data))

        for x_val in x_new
            push!(out_signal, itp_cubic(x_val))
        end


    # Data jumps somehwere
    else
        # Cubic interpolation
        x = 1:length(signal)
        itp_cubic = cubic_spline_interpolation(x, signal)

        # Fake an increased density in between the points of interest
        x_new = vcat(1:jump_idx[1], range(jump_idx[1], jump_idx[1]+1, padding)[2:end-1], (jump_idx[1]+1):length(signal))

        for x_val in x_new
            push!(out_signal, itp_cubic(x_val))
        end

    end

    return out_signal
    
end

function complete_link(signal, padding)

    if abs(signal[end] - signal[1]) > 0.01

        final_audio = Float64[]

        # Cut the data in half
        cut_location = floor(Int, length(signal)/2)
        cut_data = vcat(signal[cut_location+1:end], signal[1:cut_location])

        # Cubic interpolation
        x = 1:length(cut_data)
        itp_cubic = cubic_spline_interpolation(x, cut_data)

        # Fake an increased density in between the points of interest
        x_new = vcat(1:cut_location+1, range(cut_location+1, cut_location+2, padding)[2:end-1], (cut_location+2):length(cut_data))

        for x_val in x_new
            push!(final_audio, itp_cubic(x_val))
        end

        return final_audio
    else
        return signal
    end

end

function moving_average(A::AbstractArray, m::Int)
    out = similar(A)
    R = CartesianIndices(A)
    Ifirst, Ilast = first(R), last(R)
    I1 = m÷2 * oneunit(Ifirst)
    for I in R
        n, s = 0, zero(eltype(out))
        for J in max(Ifirst, I-I1):min(Ilast, I+I1)
            s += A[J]
            n += 1
        end
        out[I] = s/n
    end
    return out
end


##
global const bpm = Ref{Float64}(120.0)
global const spf = Ref{Int64}(256) 
global const fs = Ref{Int64}(44100)

##
starlings = VideoIO.load("starlings.mp4")
clipped_starlings = starlings[1:600]

rough_alg = AdaptiveThreshold(window_size = 4; percentage = 1)
fine_alg = AdaptiveThreshold(window_size = 200; percentage = 0.01)

# Initialize audio signal storage
all_audio = Vector{Float64}[]

# Loop over every frame
for (i, raw_img) in enumerate(clipped_starlings)
    first_blur = imfilter(raw_img, Kernel.gaussian(1))
    img = binarize(first_blur, rough_alg)
    second_blur = imfilter(img, Kernel.gaussian(12))
    second_raster = binarize(second_blur, fine_alg)

    image_values = [float(second_raster[j, i].val) for j in axes(second_raster, 1), i in axes(second_raster, 2)]

    x_vec = 1:size(image_values, 1)
    y_vec = 1:size(image_values, 2)

    image_values .= reverse(image_values, dims=2)

    try
        (xs, ys) = flock_outline(x_vec, y_vec, image_values) # coordinates of this line segment
    catch e
        @warn "Could not extract flock outline for frame $i: $e"
        continue
    end

    (xs, ys) = flock_outline(x_vec, y_vec, image_values) # coordinates of this line segment

    audio_signal = flock_deviations(xs, ys)

    # Do a course phase alignment 
    global_idx = findmax(ys)[2]
    phase_aligned_audio = vcat(audio_signal[global_idx:end], audio_signal[1:global_idx-1])

    push!(all_audio, phase_aligned_audio)
end

## Process audio signal

# Scale the audio for volume
final_audio = Float64[]
largest_amplitude = maximum(signal -> maximum(abs.(signal)), all_audio)
scaled_audio = [sign.(signal) .* (abs.(signal) ./ largest_amplitude).^(1.0) for signal in all_audio]


# Phase align properly
aligned_audio = Vector{Float64}[]
push!(aligned_audio, scaled_audio[1])
for (i, signal) in enumerate(scaled_audio[2:end])
    aligned_signal = phase_matching(signal, aligned_audio[i], 200)
    push!(aligned_audio, aligned_signal)
end

# Fill in any remaining bald spots
smooth_data = Vector{Float64}[]
for signal in aligned_audio
    filled_signal = fill_bald_spots!(signal, 200)
    push!(smooth_data, filled_signal)
end


# aligned_audio = Vector{Float64}[]
# push!(aligned_audio, moving_average(scaled_audio[1], 100))
# for (i, signal) in enumerate(scaled_audio[2:end])
#     circular_signal = moving_average(signal, 100)
#     aligned_signal = phase_matching(circular_signal, aligned_audio[i], 200)
#     push!(aligned_audio, aligned_signal)
# end



# Interpolate to fixed length
fps = 30.0
samples_per_frame = floor(Int, fs[] / fps)

for signal in smooth_data

    x = range(1, samples_per_frame, length(signal))
    y = signal

    itp_cubic = cubic_spline_interpolation(x, y)

    x_new = 1:samples_per_frame

    for x_val in x_new
        push!(final_audio, itp_cubic(x_val))
    end
    
end


## Test plotting
fig = Figure()
ax = Axis(fig[1, 1])

plot_values = [309, 561]

for i in plot_values
    # lines!(ax, 1:length(aligned_audio[i]), aligned_audio[i], color = :red)
    lines!(ax, 1:length(smooth_data[i]), smooth_data[i], color = :blue)
end
# lines!(ax, 1:length(scaled_audio[23]), scaled_audio[23], color = :black)
# lines!(ax, 1:length(scaled_audio[24]), scaled_audio[24], color = :green)
# lines!(ax, 1:length(scaled_audio[24]), phase_matching(scaled_audio[24], scaled_audio[23], 50), color = :red)
display(fig)

##

running = Ref(true)
stream = PortAudioStream(0, 1; samplerate=fs[])

##
@async begin
    try
        tᵢ = 0
        buf = zeros(Float32, spf[])

        write(stream, final_audio)

        # while running[]

        #     write(stream, final_audio[tᵢ*spf[] .+ 1 : (tᵢ+1)*spf[]])
        #     tᵢ += 1

        # end

    catch e
        @error "Audio task crashed" exception=(e, catch_backtrace())
    end
end

##
running[] = false
close(stream)