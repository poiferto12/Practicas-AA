using WAV
using DSP
using FFTW
using Statistics

function extractFeatures(audioPath::String)

    y, fs = wavread(audioPath)
    fs = Int(fs)

    if size(y, 2) > 1
        y = vec(mean(y, dims=2))
    else
        y = vec(y)
    end
    y = Float64.(y)

    nFFT      = 2048
    hopLength = 512
    nMels     = 40
    nMFCC     = 13

    if length(y) < nFFT
        y = vcat(y, zeros(nFFT - length(y)))
    end

    nFrames = div(length(y) - nFFT, hopLength) + 1
    window  = DSP.hanning(nFFT)
    nBins   = div(nFFT, 2) + 1
    freqs   = collect(0:(nBins-1)) .* (fs / nFFT)

    magSpec   = zeros(nBins, nFrames)
    powerSpec = zeros(nBins, nFrames)
    zcrVec    = zeros(nFrames)
    rmsVec    = zeros(nFrames)

    for i in 1:nFrames
        iStart = (i - 1) * hopLength + 1
        iStop  = iStart + nFFT - 1
        frame  = iStop <= length(y) ? y[iStart:iStop] : vcat(y[iStart:end], zeros(iStop - length(y)))

        spec = fft(frame .* window)[1:nBins]
        magSpec[:, i]   = abs.(spec)
        powerSpec[:, i] = abs.(spec) .^ 2

        signs = sign.(frame)
        signs[signs .== 0.0] .= 1.0
        zcrVec[i] = sum(abs.(diff(signs))) / (2.0 * nFFT)
        rmsVec[i] = sqrt(mean(frame .^ 2))
    end

    centroidVec  = zeros(nFrames)
    bandwidthVec = zeros(nFrames)
    for i in 1:nFrames
        mag = magSpec[:, i]
        s   = sum(mag)
        if s > 0.0
            centroidVec[i]  = sum(freqs .* mag) / s
            bandwidthVec[i] = sqrt(sum(((freqs .- centroidVec[i]) .^ 2) .* mag) / s)
        end
    end

    rolloffVec = zeros(nFrames)
    for i in 1:nFrames
        power      = powerSpec[:, i]
        totalPower = sum(power)
        if totalPower > 0.0
            cumPow        = cumsum(power)
            idx           = findfirst(cumPow .>= 0.85 * totalPower)
            rolloffVec[i] = idx === nothing ? freqs[end] : freqs[idx]
        end
    end

    hz2mel(f) = 2595.0 * log10(1.0 + f / 700.0)
    mel2hz(m) = 700.0 * (10.0 ^ (m / 2595.0) - 1.0)

    melPoints = range(hz2mel(0.0), hz2mel(fs / 2.0), length = nMels + 2)
    hzPoints  = mel2hz.(melPoints)
    binPoints = clamp.(floor.(Int, hzPoints .* (nFFT / fs)) .+ 1, 1, nBins)

    melFilters = zeros(nMels, nBins)
    for m in 1:nMels
        lo = binPoints[m]; pk = binPoints[m+1]; hi = binPoints[m+2]
        for k in lo:pk
            melFilters[m, k] = pk > lo ? (k - lo) / (pk - lo) : 1.0
        end
        for k in pk:hi
            melFilters[m, k] = hi > pk ? (hi - k) / (hi - pk) : 1.0
        end
    end

    logMelEnergies = log.(max.(melFilters * powerSpec, 1e-10))

    mfccMatrix = zeros(nMFCC, nFrames)
    for n in 1:nMFCC
        for m in 1:nMels
            mfccMatrix[n, :] .+= logMelEnergies[m, :] .* cos(π * (n - 1) * (m - 0.5) / nMels)
        end
        mfccMatrix[n, :] .*= sqrt(2.0 / nMels)
    end

    features = Float64[]
    sizehint!(features, 36)

    for n in 1:nMFCC
        push!(features, mean(mfccMatrix[n, :]), std(mfccMatrix[n, :]))
    end
    push!(features, mean(zcrVec),       std(zcrVec))
    push!(features, mean(rmsVec),       std(rmsVec))
    push!(features, mean(centroidVec),  std(centroidVec))
    push!(features, mean(bandwidthVec), std(bandwidthVec))
    push!(features, mean(rolloffVec),   std(rolloffVec))

    @assert length(features) == 36

    return features
end

function loadDataset(datasetFolder::String)
    catFiles = filter(f -> endswith(f, ".wav"), readdir(joinpath(datasetFolder, "cats"), join=true))
    dogFiles = filter(f -> endswith(f, ".wav"), readdir(joinpath(datasetFolder, "dogs"), join=true))
    cowFiles = filter(f -> endswith(f, ".wav"), readdir(joinpath(datasetFolder, "Cow"), join=true))
    frogFiles = filter(f -> endswith(f, ".wav"), readdir(joinpath(datasetFolder, "frogs"), join=true))

    println("Cargando dataset: $(length(catFiles)) gatos, $(length(dogFiles)) perros, $(length(cowFiles)) vacas, $(length(frogFiles)) ranas...")

    catFeatures = extractFeatures.(catFiles)
    dogFeatures = extractFeatures.(dogFiles)
    cowFeatures = extractFeatures.(cowFiles)
    frogFeatures = extractFeatures.(frogFiles)

    inputs  = Float32.(collect(hcat([catFeatures; dogFeatures; cowFeatures; frogFeatures]...)'))
    targets = [repeat(["cats"], length(catFiles)); repeat(["dogs"], length(dogFiles)); repeat(["cows"], length(cowFiles)); repeat(["frogs"], length(frogFiles))]

    println("Tamaño de la matriz de entradas: ", size(inputs, 1), "x", size(inputs, 2), " de tipo ", typeof(inputs))
    println("Clases: gatos ($(length(catFiles))), perros ($(length(dogFiles))), vacas ($(length(cowFiles))), ranas ($(length(frogFiles)))")
    return inputs, targets
end
