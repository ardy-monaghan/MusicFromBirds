using BlobTracking, Images, VideoIO

## Get the tracks

path = "./bird_data/feb26/raw_swipe.MOV"
io   = VideoIO.open(path)
vid  = VideoIO.openvideo(io)
img  = first(vid)

# path = "./bird_data/feb26/test.mov"
# io   = VideoIO.open(path)
# vid  = VideoIO.openvideo(io)
# img  = first(vid)

medbg = MedianBackground(Float32.(img), 100) # A buffer of 100 frames
foreach(1:100) do i # Populate the buffer
    update!(medbg, Float32.(first(vid)))
end
bg = background(medbg)

Gray.(bg)

mask = ones(size(bg))
mask[:,1100:end] .= 0
mask[1:200,:] .= 0

Gray.(mask)

function preprocessor(storage, img)
    storage .= Float32.(img)
    update!(medbg, storage) # update the background model
    storage .= Float32.(abs.(storage .- background(medbg)) .> 0.25) # You can save some computation by not calculating a new background image every sample
end

test_img = copy(img)
preprocessor(test_img, img)
test_img


bt = BlobTracker(5:10, #sizes 
                2.0, # σw Dynamics noise std.
                10.0,  # σe Measurement noise std. (pixels)
                mask=mask,
                preprocessor = preprocessor,
                amplitude_th = 0.05,
                correspondence = HungarianCorrespondence(p=1.0, dist_th=2), # dist_th is the number of sigmas away from a predicted location a measurement is accepted.
)

result = track_blobs(bt, vid,
                         display = nothing, # use nothing to omit displaying.
                         recorder = Recorder()) # records result to video on disk


## Plot the tracks

traces = trace(result, minlife=5) # Filter minimum lifetime of 5
measurement_traces = tracem(result, minlife=5)
drawimg = RGB.(img)
draw!(drawimg, traces, c=RGB(0,0,0.5))
draw!(drawimg, measurement_traces, c=RGB(0.5,0,0))

## Get the midi notes 

# Loop over each time
# Check if a point was previously to the left, and now to the right
# Find which note to play (mod the y position)
# Push midi note