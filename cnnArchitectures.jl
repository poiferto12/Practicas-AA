using Flux

# ============================================================
# CNN 1
# ============================================================

function buildCnn1(numClasses)

    Chain(

        Conv((3,1), 1=>8, relu, pad=(1,0)),
        MaxPool((2,1)),

        Flux.flatten,

        Dense(16384, numClasses),

        softmax
    )

end

# ============================================================
# CNN 2
# ============================================================

function buildCnn2(numClasses)

    Chain(

        Conv((3,1), 1=>8, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 8=>16, relu, pad=(1,0)),
        MaxPool((2,1)),

        Flux.flatten,

        Dense(16384, numClasses),

        softmax
    )

end

# ============================================================
# CNN 3
# ============================================================

function buildCnn3(numClasses)

    Chain(

        Conv((3,1), 1=>8, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 8=>16, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 16=>32, relu, pad=(1,0)),
        MaxPool((2,1)),

        Flux.flatten,

        Dense(16384, numClasses),

        softmax
    )

end

# ============================================================
# CNN 4
# ============================================================

function buildCnn4(numClasses)

    Chain(

        Conv((3,1), 1=>16, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 16=>32, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 32=>64, relu, pad=(1,0)),
        MaxPool((2,1)),

        Flux.flatten,

        Dense(32768, 128),
        relu,

        Dense(128, numClasses),

        softmax
    )

end

# ============================================================
# CNN 5
# ============================================================

function buildCnn5(numClasses)

    Chain(

        Conv((3,1), 1=>16, relu, pad=(1,0)),
        MaxPool((2,1)),
        Dropout(0.25),

        Conv((3,1), 16=>32, relu, pad=(1,0)),
        MaxPool((2,1)),
        Dropout(0.30),

        Conv((3,1), 32=>64, relu, pad=(1,0)),
        MaxPool((2,1)),
        Dropout(0.40),

        Flux.flatten,

        Dense(32768, 128),
        relu,

        Dropout(0.40),

        Dense(128, numClasses),

        softmax
    )

end

# ============================================================
# CNN 6
# ============================================================

function buildCnn6(numClasses)

    Chain(

        Conv((3,1), 1=>16, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 16=>32, relu, pad=(1,0)),
        MaxPool((2,1)),

        Flux.flatten,

        Dense(32768, 64),
        relu,

        Dense(64, numClasses),

        softmax
    )

end

# ============================================================
# CNN 7
# ============================================================

function buildCnn7(numClasses)

    Chain(

        Conv((3,1), 1=>4, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 4=>8, relu, pad=(1,0)),
        MaxPool((2,1)),

        Flux.flatten,

        Dense(8192, numClasses),

        softmax
    )

end

# ============================================================
# CNN 8
# ============================================================

function buildCnn8(numClasses)

    Chain(

        Conv((3,1), 1=>8, relu, pad=(1,0)),
        BatchNorm(8),
        MaxPool((2,1)),

        Conv((3,1), 8=>16, relu, pad=(1,0)),
        BatchNorm(16),
        MaxPool((2,1)),

        Conv((3,1), 16=>32, relu, pad=(1,0)),
        BatchNorm(32),
        MaxPool((2,1)),

        Flux.flatten,

        Dense(16384, numClasses),

        softmax
    )

end

# ============================================================
# CNN 9
# ============================================================

function buildCnn9(numClasses)

    Chain(

        Conv((3,1), 1=>16, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 16=>32, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 32=>64, relu, pad=(1,0)),
        MaxPool((2,1)),

        Conv((3,1), 64=>128, relu, pad=(1,0)),
        MaxPool((2,1)),

        Flux.flatten,

        Dense(32768, 256),
        relu,

        Dense(256, numClasses),

        softmax
    )

end

# ============================================================
# CNN 10
# ============================================================

function buildCnn10(numClasses)

    Chain(

        Conv((3,1), 1=>4, relu, pad=(1,0)),
        MaxPool((2,1)),

        Flux.flatten,

        Dense(8192, numClasses),

        softmax
    )

end