println("==========================================================")
println("   PRUEBAS COMPLETAS DE TODAS LAS FUNCIONES IMPLEMENTADAS")
println("==========================================================")

include("firmas.jl")

# Cargar datos de prueba
using DelimitedFiles: readdlm
dataset = readdlm("iris.data",',')
inputs = convert(Array{Float32,2}, dataset[:,1:4])
println("✓ Datos iris cargados: ", size(inputs))

# ==========================================================================
# ================================ oneHotEncoding ==========================
# ==========================================================================
println("\n=== PRUEBAS oneHotEncoding ===")

# Variante 1: con feature y classes explícitas
feature1 = ["A", "B", "A", "C", "B", "A"]
classes1 = ["A", "B", "C"]
result1 = oneHotEncoding(feature1, classes1)
println("✓ oneHotEncoding(feature, classes): ", size(result1))
println("  Ejemplo: feature=[A,B,A,C,B,A] -> shape=", size(result1))

# Variante 2: solo con feature (classes automáticas)
feature2 = [1, 2, 1, 3, 2, 1]
result2 = oneHotEncoding(feature2)
println("✓ oneHotEncoding(feature): ", size(result2))
println("  Ejemplo: feature=[1,2,1,3,2,1] -> shape=", size(result2))

# Variante 3: con array booleano
feature3 = [true, false, true, false, true]
result3 = oneHotEncoding(feature3)
println("✓ oneHotEncoding(Bool[]): ", size(result3))
println("  Ejemplo: feature=[T,F,T,F,T] -> shape=", size(result3))

# Caso binario (≤2 clases)
feature_binary = ["Si", "No", "Si", "No"]
result_binary = oneHotEncoding(feature_binary)
println("✓ oneHotEncoding binario: ", size(result_binary))

# Caso con datos iris
iris_targets = oneHotEncoding(dataset[:,5])
println("✓ oneHotEncoding iris: ", size(iris_targets))

# ==========================================================================
# ========================= Parámetros de normalización ===================
# ==========================================================================
println("\n=== PRUEBAS calculateNormalizationParameters ===")

# Parámetros MinMax
minmax_params = calculateMinMaxNormalizationParameters(inputs)
println("✓ calculateMinMaxNormalizationParameters: ")
println("  Min: ", round.(minmax_params[1], digits=2))
println("  Max: ", round.(minmax_params[2], digits=2))

# Parámetros ZeroMean
zeromean_params = calculateZeroMeanNormalizationParameters(inputs)
println("✓ calculateZeroMeanNormalizationParameters: ")
println("  Media: ", round.(zeromean_params[1], digits=2))
println("  Std: ", round.(zeromean_params[2], digits=2))

# ==========================================================================
# ============================= normalizeMinMax ==========================
# ==========================================================================
println("\n=== PRUEBAS normalizeMinMax ===")

# Variante 1: normalizeMinMax! (modifica original)
test_inputs1 = copy(inputs[1:5,:])
original_range = (minimum(test_inputs1), maximum(test_inputs1))
normalizeMinMax!(test_inputs1)
new_range = (minimum(test_inputs1), maximum(test_inputs1))
println("✓ normalizeMinMax!(dataset): ")
println("  Antes: min=", round(original_range[1], digits=2), " max=", round(original_range[2], digits=2))
println("  Después: min=", round(new_range[1], digits=2), " max=", round(new_range[2], digits=2))

# Variante 2: normalizeMinMax!(dataset, params)
test_inputs2 = copy(inputs[1:5,:])
params = calculateMinMaxNormalizationParameters(test_inputs2)
normalizeMinMax!(test_inputs2, params)
println("✓ normalizeMinMax!(dataset, params): min=", round(minimum(test_inputs2), digits=2))

# Variante 3: normalizeMinMax (sin modificar original)
test_inputs3 = copy(inputs[1:5,:])
original_copy = copy(test_inputs3)
normalized_copy = normalizeMinMax(test_inputs3)
println("✓ normalizeMinMax(dataset): ")
println("  Original sin cambios: ", test_inputs3 == original_copy)
println("  Copia normalizada: min=", round(minimum(normalized_copy), digits=2), " max=", round(maximum(normalized_copy), digits=2))

# ==========================================================================
# ============================ normalizeZeroMean =======================
# ==========================================================================
println("\n=== PRUEBAS normalizeZeroMean ===")

# Variante 1: normalizeZeroMean! (modifica original)
test_inputs4 = copy(inputs[1:10,:])
original_stats = (mean(test_inputs4), std(test_inputs4))
normalizeZeroMean!(test_inputs4)
new_stats = (mean(test_inputs4), std(test_inputs4))
println("✓ normalizeZeroMean!(dataset): ")
println("  Antes: media=", round(original_stats[1], digits=3), " std=", round(original_stats[2], digits=3))
println("  Después: media=", round(new_stats[1], digits=3), " std=", round(new_stats[2], digits=3))

# Variante 2: normalizeZeroMean!(dataset, params)
test_inputs5 = copy(inputs[1:10,:])
params2 = calculateZeroMeanNormalizationParameters(test_inputs5)
normalizeZeroMean!(test_inputs5, params2)
println("✓ normalizeZeroMean!(dataset, params): media≈0=", abs(mean(test_inputs5)) < 0.01)

# Variante 3: normalizeZeroMean (sin modificar original)
test_inputs6 = copy(inputs[1:10,:])
original_copy2 = copy(test_inputs6)
normalized_copy2 = normalizeZeroMean(test_inputs6)
original_unchanged = test_inputs6 == original_copy2
println("✓ normalizeZeroMean(dataset): ")
if !original_unchanged
    println("  ⚠️  BUG DETECTADO: Original fue modificado (debería usar 'datasets' no 'dataset')")
else
    println("  Original sin cambios: ", original_unchanged)
end
println("  Copia normalizada: media≈0=", abs(mean(normalized_copy2)) < 0.01)

# ==========================================================================
# ================================ accuracy ===============================
# ==========================================================================
println("\n=== PRUEBAS accuracy ===")

# Preparar datos de prueba
bool_outputs1 = [true, false, true, false, true]
bool_targets1 = [true, false, false, false, true]

# Variante 1: accuracy(Bool[], Bool[]) - FUNCIONA
acc1 = accuracy(bool_outputs1, bool_targets1)
println("✓ accuracy(Bool[], Bool[]): ", acc1)

# Variante 2: accuracy(Bool[2D], Bool[2D]) - FUNCIONA
bool_outputs2 = [true false; false true; true false]
bool_targets2 = [true false; false true; false true]
acc2 = accuracy(bool_outputs2, bool_targets2)
println("✓ accuracy(Bool[2D], Bool[2D]): ", acc2)

# Caso especial: matriz de 1 columna (binaria)
single_col_out = reshape([true, false, true], 3, 1)
single_col_targ = reshape([true, false, false], 3, 1)  
acc_single_col = accuracy(single_col_out, single_col_targ)
println("✓ accuracy(Bool[2D] 1 col): ", acc_single_col)

# Caso multiclase (>2 columnas)
multi_outputs = [true false false; false true false; false false true]
multi_targets = [true false false; false true false; true false false]
acc_multi = accuracy(multi_outputs, multi_targets)
println("✓ accuracy(Bool[2D] multiclase): ", acc_multi)

println("\n--- accuracy con Real[] - REQUIERE classifyOutputs (no implementada) ---")
try
    real_outputs1 = [0.8, 0.2, 0.9, 0.1, 0.7]
    acc3 = accuracy(real_outputs1, bool_targets1)
    println("✓ accuracy(Real[], Bool[]): ", acc3)
catch e
    println("❌ accuracy(Real[], Bool[]): Requiere classifyOutputs")
end

try
    real_outputs2 = [0.8 0.2; 0.1 0.9; 0.7 0.3]
    acc4 = accuracy(real_outputs2, bool_targets2)
    println("✓ accuracy(Real[2D], Bool[2D]): ", acc4)
catch e
    println("❌ accuracy(Real[2D], Bool[2D]): Requiere classifyOutputs")
end

# ==========================================================================
# ============================== buildClassANN ===========================
# ==========================================================================
println("\n=== PRUEBAS buildClassANN ===")

# Caso 1: Red sin capas ocultas, binaria
ann1 = buildClassANN(4, Int[], 2)
println("✓ buildClassANN(4, [], 2): ", length(ann1), " capas")

# Caso 2: Red con 1 capa oculta, binaria
ann2 = buildClassANN(4, [5], 2)
println("✓ buildClassANN(4, [5], 2): ", length(ann2), " capas")

# Caso 3: Red con múltiples capas ocultas, multiclase
ann3 = buildClassANN(4, [10, 5], 3)
println("✓ buildClassANN(4, [10,5], 3): ", length(ann3), " capas")

# Caso 4: Con funciones de transferencia personalizadas
using Flux: relu, tanh
custom_functions = [relu, tanh]
ann4 = buildClassANN(4, [8, 6], 3, transferFunctions=custom_functions)
println("✓ buildClassANN con funciones personalizadas: ", length(ann4), " capas")

# Caso 5: Muchas clases
ann5 = buildClassANN(10, [20, 15, 10], 10)
println("✓ buildClassANN(10, [20,15,10], 10): ", length(ann5), " capas")

# Probar predicciones
test_input = randn(Float32, 4, 5)  # 4 features, 5 muestras
pred1 = ann1(test_input)
pred2 = ann2(test_input)  
pred3 = ann3(test_input)

println("✓ Predicción ann1 (binaria): ", size(pred1))
println("✓ Predicción ann2 (binaria): ", size(pred2))
println("✓ Predicción ann3 (multiclase): ", size(pred3))

# Verificar que softmax da probabilidades
println("✓ Suma probabilidades ann3: ", sum(pred3, dims=1))

# ==========================================================================
# ========================== CASOS ESPECIALES ============================
# ==========================================================================
println("\n=== PRUEBAS DE CASOS ESPECIALES ===")

# oneHotEncoding con una sola clase
single_class = ["A", "A", "A", "A"]
result_single = oneHotEncoding(single_class)
println("✓ oneHotEncoding clase única: ", size(result_single))

# Normalización con valores idénticos 
identical_data = ones(Float32, 5, 3)
normalize_identical = normalizeMinMax(identical_data)
println("✓ Normalización datos idénticos: ", extrema(normalize_identical))

# Red neuronal con 1 sola capa oculta y 1 neurona
tiny_ann = buildClassANN(2, [1], 2)
println("✓ Red mínima: ", length(tiny_ann), " capas")

println("\n==========================================================")
println("              ✓ TODAS LAS PRUEBAS COMPLETADAS")
println("==========================================================")

# Mostrar resumen de funciones probadas
println("\n=== FUNCIONES PROBADAS ===")
println("• oneHotEncoding (3 variantes)")
println("• calculateMinMaxNormalizationParameters")
println("• calculateZeroMeanNormalizationParameters") 
println("• normalizeMinMax (2 variantes implementadas)")
println("• normalizeMinMax! (2 variantes)")
println("• normalizeZeroMean (1 variante implementada)")
println("• normalizeZeroMean! (2 variantes)")
println("• accuracy (4 variantes)")
println("• buildClassANN (múltiples configuraciones)")

println("\n=== FUNCIONES NO IMPLEMENTADAS (tienen solo esqueletos) ===")

# Probar qué devuelven las funciones no implementadas
println("classifyOutputs([0.1, 0.9]) devuelve:", classifyOutputs([0.1, 0.9]))
println("holdOut(100, 0.2) devuelve:", holdOut(100, 0.2))  
println("confusionMatrix([true,false], [true,true]) devuelve:", confusionMatrix([true,false], [true,true]))

println("\n=== LISTA DE FUNCIONES NO IMPLEMENTADAS ===")
println("• classifyOutputs (2 variantes) - devuelve nothing")
println("• normalizeMinMax con parámetros (1 variante)")  
println("• normalizeZeroMean con parámetros (1 variante)")
println("• trainClassANN (4 variantes) - devuelve nothing")
println("• holdOut (2 variantes) - devuelve nothing")
println("• confusionMatrix (6 variantes) - devuelve nothing") 
println("• crossvalidation (4 variantes) - devuelve nothing")
println("• ANNCrossValidation - devuelve nothing")
println("• modelCrossValidation - devuelve nothing")
println("• trainClassDoME (3 variantes) - devuelve nothing")
