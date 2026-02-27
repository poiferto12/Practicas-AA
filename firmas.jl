
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses
using Flux: σ

#feature -> vector con los valores de un atributo o salida deseada para cada patron (ven sendo targets)
#classes -> valores de las categorias

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    unique_classes = unique(classes);
    numClasses = length(unique_classes);
    if numClasses<=2
        # Si solo hay dos clases, se genera una matriz con una columna
        oneHot = reshape(feature.==unique_classes[1], :, 1);
    else
        # Si hay mas de dos clases se genera una matriz con una columna por clase

        oneHot = BitArray{2}(undef, size(feature,1), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==unique_classes[numClass]);
        end;

        #### Puxen o sin bucles porque me parecia mas facil de entender, despos xa miramos cal renta mas ####

        # Una forma de hacerlo sin bucles sería la siguiente:
        # oneHot = convert(BitArray{2}, hcat([instance.==classes for instance in targets]...)');
        # targets = oneHot;
    end;
    return oneHot;
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
### Esta funcion nn tou seuguro de que te ben, pide non usar function e esto é o unico que lle vexo sentido e non me salta errorores xd, despos proboa ###

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1);
    ### Ven sendo usar o de arriba, pero quitando o unique_classes ###
end;

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    normalizationParameters = (minimum(dataset, dims=1), maximum(dataset, dims=1));
    return normalizationParameters;
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    normalizationParameters = (mean(dataset, dims=1), std(dataset, dims=1));
    return normalizationParameters;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters;
    # Se modifican los datos originales
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    minValues, maxValues = calculateMinMaxNormalizationParameters(dataset);
    # Se modifican los datos originales
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters;
    # Copy para no modificar los datos originales
    dataset = copy(dataset);
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    minValues, maxValues = calculateMinMaxNormalizationParameters(dataset);
    # Copy para no modificar los datos originales
    dataset = copy(dataset);
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues, stdValues = normalizationParameters;
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    avgValues, stdValues = calculateZeroMeanNormalizationParameters(dataset);
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues, stdValues = normalizationParameters;
    # Copy para no modificar los datos originales
    dataset = copy(dataset);
    dataset .-= avgValues;
    dataset ./= stdValues;
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    avgValues, stdValues = calculateZeroMeanNormalizationParameters(dataset);
    # Copy para no modificar los datos originales
    dataset = copy(dataset);
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    # Devuelve un vector booleano: true si output es >= que threshold y false si no lo es
    return outputs .>=threshold 
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    ### Ahora deberia de funcionar correcto segun o que se pide no enunciado ###
    if(size(outputs,2) == 1);
        outputs = (outputs .>= threshold);
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
    end;
    return outputs;
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Calcula la proporción de aciertos comparando salidas y objetivos
    return mean(targets .== outputs);
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    # Calcula la accuracy para salidas booleanas en forma matricial
    if size(outputs,2) == 1
        # Caso binario
        return accuracy(outputs[:,1], targets[:,1]);
    elseif size(outputs, 2) > 2
        # Caso multiclase
        return mean(all(outputs .== targets, dims=2));
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Convierte salidas reales a booleanas usando un umbral y calcula la accuracy
    binary_outputs = classifyOutputs(outputs, threshold=threshold);
    return accuracy(binary_outputs, targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    # Calcula la accuracy para salidas reales en formato matricial
    if size(outputs,2) == 1
        # Caso binario
        return accuracy(outputs[:,1], targets[:,1], threshold=threshold);
    else
        # Caso multiclase
        binary_outputs = classifyOutputs(outputs, threshold=threshold);
        return accuracy(binary_outputs, targets);
    end;
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain();
    numInputsLayer = numInputs;
    i = 1;
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]));
        numInputsLayer = numOutputsLayer;
        i += 1;
    end; 
    if (numOutputs > 2); 
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);        
    else        ### Ahora funciona ben con 2 ou menos. i ao salir do bucle tiña leght(topology) + 1 como valor e transferFunctions[] solo ten length(top..) elementos, de ahi o Boundserroror, intentaba acceder ao 4 cando solo tiña 3 elementos ###
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    end;
    return ann;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    inputs = Float32.(dataset[1]);
    targets = dataset[2];
    ann = buildClassANN(size(inputs, 2), topology, size(targets, 2); transferFunctions);

    ### A funcion "loss" é pa decidir que funcion usar para entrenar a RNA, recibe as salidas do modelo e as salidas deseadas ###
    ### Cambiei a forma de elegir loss a esta que encontrei, asi deberiamos de poder evitar o erroror de antes ###
    loss(x,y) = (size(targets, 2) == 1) ? 
        (Losses.binarycrossentropy(ann(x),y)) : 
        (Losses.crossentropy(ann(x),y));

    ###Este é o vector cos valores de loss en cada ciclo ###
    lossValues = zeros(Float32, maxEpochs+1);
    lossValues[1] = loss(inputs', targets');

    ### Parece que habia que crear o Adam 1 sola vez, antes faciase en cada epoch ###
    opt = Flux.Optimise.Adam(learningRate);
    currentEpoch = 1;
    while (currentEpoch <= maxEpochs) && (lossValues[currentEpoch] > minLoss);
        Flux.train!(loss, Flux.params(ann), [(inputs', targets')], opt);
        lossValues[currentEpoch+1] = loss(inputs', targets');
        currentEpoch += 1;
    end;
    return (ann, lossValues);
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    return trainClassANN(topology, (inputs, reshape(targets, (:,1))); transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    @assert (P >= 0.0) && (P <= 1.0) "P debe estar entre 0 y 1";
    # Permutación aleatoria de los índices 1..N
    indices = randperm(N);
    # Número de patrones de entrenamiento
    numTrainingInstances = Int(round(N * (1 - P)));
    # Separamos los índices en entrenamiento y test
    trainingIdx = indices[1:numTrainingInstances];
    testIdx = indices[numTrainingInstances+1:end];
    return trainingIdx, testIdx;
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    # Test
    trainingValIdx, testIdx = holdOut(N, Ptest)
    # Calculamos la proporción ajustada para validación
    nTrainingVal = length(trainingValIdx)
    # Número de patrones de entrenamiento
    nVal = Int(round(N * Pval))
    PvalAdjusted = nVal / nTrainingVal
    # Validación
    trainingIdx, valIdx = holdOut(nTrainingVal, PvalAdjusted)
    # Mapeamos los índices de trainingIdx y valIdx al original
    trainingIdx = trainingValIdx[trainingIdx]
    valIdx = trainingValIdx[valIdx]
    return(trainingIdx, valIdx, testIdx);
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
   
    inputs, targets = Float32.(trainingDataset[1]), trainingDataset[2];
    validation_inputs, validation_targets = Float32.(validationDataset[1]), validationDataset[2];
    test_inputs, test_targets = Float32.(testDataset[1]), testDataset[2];

    # Comprobar si están vacíos
    validation_empty = size(validation_inputs, 1) == 0;
    test_empty = size(test_inputs, 1) == 0;

    ann = buildClassANN(size(inputs, 2), topology, size(targets, 2); transferFunctions);

    ### A funcion "loss" é pa decidir que funcion usar para entrenar a RNA, recibe as salidas do modelo e as salidas deseadas ###
    ### Cambiei a forma de elegir loss a esta que encontrei, asi deberiamos de poder evitar o erroror de antes ###
    loss(x,y) = (size(targets, 2) == 1) ? 
        (Losses.binarycrossentropy(ann(x),y)) : 
        (Losses.crossentropy(ann(x),y));

    trainLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];

    # Cálculo loss ciclo 0
    push!(trainLosses, Float32(loss(inputs', targets')));

    # Si validation_empty, validationLosses se queda vacío
    if !validation_empty
        push!(validationLosses, Float32(loss(validation_inputs', validation_targets')));
    end

    # Si test_empty, testLosses se queda vacío
    if !test_empty
        push!(testLosses, Float32(loss(test_inputs', test_targets')));
    end

    best_ann = deepcopy(ann);
    best_validationLoss = validation_empty ? Inf : validationLosses[1];
    nEpochs_without_improve = 0;
    currentEpoch = 1;

    ### Parece que habia que crear o Adam 1 sola vez, antes faciase en cada epoch ###
    opt = Flux.Optimise.Adam(learningRate);

    while currentEpoch <= maxEpochs
        # Training de un ciclo
        Flux.train!(loss, Flux.params(ann), [(inputs', targets')], opt);

        # Cálculo loss de training
        current_trainLoss = Float32(loss(inputs', targets'));
        push!(trainLosses, current_trainLoss)
        
        # Cálculo loss de validacion
        if !validation_empty
            current_validationLoss = Float32(loss(validation_inputs', validation_targets'));
            push!(validationLosses, current_validationLoss);
            
            # Comprueba si hay mejora de validación
            if current_validationLoss < best_validationLoss
                best_validationLoss = current_validationLoss
                best_ann = deepcopy(ann)
                nEpochs_without_improve = 0
            else
                nEpochs_without_improve += 1
            end
        end
        
        # Cálculo loss de test
        if !test_empty
            current_testLoss = Float32(loss(test_inputs', test_targets'));
            push!(testLosses, current_testLoss);
        end

        # Para por alcanzar la pérdida mínima
        if current_trainLoss <= minLoss
            break
        end

        # Parada temprana por validación
        if !validation_empty && nEpochs_without_improve >= maxEpochsVal
            break
        end

        currentEpoch += 1
    end

    if !validation_empty
        ann = best_ann  # Se devuelve la mejor RNA según validación
    end

    return (ann, trainLosses, validationLosses, testLosses);


end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    ### Fixen esto asi porque non teño muy claro se o que estaba intentando poñer antes estaba ben ou se era una sachada jajaj###
    return trainClassANN(topology, (trainingDataset[1], reshape(trainingDataset[2], :, 1));
        validationDataset = (validationDataset[1], reshape(validationDataset[2], :, 1)),
        testDataset = (testDataset[1], reshape(testDataset[2], :, 1)),
        transferFunctions = transferFunctions,
        maxEpochs = maxEpochs,
        minLoss = minLoss,
        learningRate = learningRate,
        maxEpochsVal = maxEpochsVal);
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Comprobamos dimensiones
    @assert size(outputs,2) == size(targets,2) "Las matrices deben tener el mismo número de columnas"
    numClasses = size(outputs,2)
    @assert numClasses!= 2 "No válido para matrices de dos columnas"
    if numClasses == 1
        return confusionMatrix(vec(outputs),vec(targets))
    end
    # Reservamos memoria para las métricas
    sensibilidad = zeros(numClasses)
    especificidad = zeros(numClasses)
    VPP = zeros(numClasses)
    VPN = zeros(numClasses)
    F1 = zeros(numClasses)
    # Calculamos métricas por clase
    for c in 1:numClasses
        acc, error, sen, esp, vpp, vpn, f1, _ = confusionMatrix(outputs[:,c], targets[:,c])
        sensibilidad[c]= sen
        especificidad[c]= esp
        VPP[c]= vpp
        VPN[c]= vpn
        F1[c]= f1
    end
    # Matriz de confusión
    MatrizConfusion = targets' * outputs
    # Estrategia macro (weighted)
    if weighted
        weights= vec(sum(targets, dims=1))
        sensibilidadMedia= sum(sensibilidad .* weights) / sum(weights)
        especificidadMedia = sum(especificidad .* weights) / sum(weights)
        VPPMedia= sum(VPP .* weights) / sum(weights)
        VPNMedia= sum(VPN .* weights) / sum(weights)
        F1Media = sum(F1 .* weights) / sum(weights)
    else
        sensibilidadMedia =mean(sensibilidad)
        especificidadMedia =mean(especificidad)
        VPPMedia = mean(VPP)
        VPNMedia= mean(VPN)
        F1Media = mean(F1)
    end
    # Calculamos precisión y tasa de error
    acc= accuracy(outputs, targets)
    error= 1 - acc
    return (acc,error,sensibilidadMedia,especificidadMedia,VPPMedia,VPNMedia,F1Media,MatrizConfusion)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    # Convertimos outputs a booleano
    outputsBool= classifyOutputs(outputs; threshold=threshold)
    return confusionMatrix(outputsBool,targets; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Comprobamos que todas las etiquetas están en classes
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]))
    # Codificamos outputs y targets
    outputsBool= oneHotEncoding(outputs,classes)
    targetsBool= oneHotEncoding(targets,classes)
    return confusionMatrix(outputsBool,targetsBool; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Sacamos la lista de clases que aparecen en outputs y targets
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs,targets,classes; weighted=weighted)
end;

using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    # Convertimos las entradas de entrenamiento y test a float 
    trainingInputs= Float64.(trainingDataset[1])
    trainingTargets= trainingDataset[2]
    testInputs= Float64.(testInputs)
    # Entrenamos el modelo DoME 
    model, _, _, _ = dome(trainingInputs, trainingTargets; maximumNodes=maximumNodes)
    # Evaluamos el modelo en el conjunto de test
    testOutputs = evaluateTree(model, testInputs)
    # Si la salida es un valor constante entonces repite ese valor para todas las filas
    if isa(testOutputs,Real)
        testOutputs = repeat([testOutputs],size(testInputs,1))
    end
    return testOutputs
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    # Convertimos las entradas de entrenamiento y test a Float64
    trainingInputs= Float64.(trainingDataset[1])
    trainingTargets= trainingDataset[2]
    testInputs= Float64.(testInputs)
    numClasses= size(trainingTargets,2)
    # Si solo hay una clase entonces se llama a la versión binaria y devolvemos una matriz columna
    if numClasses == 1
        return reshape(trainClassDoME((trainingInputs,vec(trainingTargets)),testInputs,maximumNodes), :, 1)
    end
    # Creamos una matriz para guardar las salidas de test para cada clase
    testOutputs = Array{Float64,2}(undef,size(testInputs,1),numClasses)
    # Para cada clase entrena un modelo DOME usando la columna correspondiente de targets
    for c in 1:numClasses
        testOutputs[:,c] = trainClassDoME((trainingInputs, trainingTargets[:,c]), testInputs, maximumNodes)
    end
    return testOutputs
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    # Convertimos las entradas de entrenamiento y test a float
    trainingInputs= Float64.(trainingDataset[1])
    trainingTargets= trainingDataset[2]
    # Calculamos el vector de clases presentes en trainingTargets
    classes= unique(trainingTargets)
    # Preparamos el vector de salida para las predicciones de test
    testOutputs= Array{eltype(trainingTargets),1}(undef,size(testInputs,1))
    testOutputsDoME= trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes)
    # Clasificamos las salidas numéricas en etiquetas con umbral 0
    testOutputsBool= classifyOutputs(testOutputsDoME; threshold=0)
    if length(classes)<= 2
        # Si hay una o dos clases entonces se le asigna la etiqueta correspondiente según el valor booleano
        testOutputsBool= vec(testOutputsBool)
        testOutputs[testOutputsBool] .= classes[1]
        if length(classes) == 2
            testOutputs[.!testOutputsBool] .= classes[2]
        end
    else
        # Si hay mas de dos clases entonces para cada clase asigna la etiqueta a las instancias clasificadas en esa clase
        for numClass in 1:length(classes)
            testOutputs[testOutputsBool[:,numClass]] .= classes[numClass]
        end
    end
    return testOutputs
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    @assert k > 1 "k debe ser mayor que 1"
    @assert N >= k "Debe haber al menos k patrones"

    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N]; # Ajustamos el tamaño a N
    shuffle!(indices);

    return indices;
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    @assert sum(targets) >= k "No hay suficientes patrones positivos para k-fold"
    @assert sum(.!targets) >= k "No hay suficientes patrones negativos para k-fold"

    indices = Array{Int64,1}(undef, length(targets));

    # Positivos
    posIdx = targets
    indices[posIdx] = crossvalidation(sum(posIdx), k);

    # Negativos
    negIdx = .!targets
    indices[negIdx] = crossvalidation(sum(negIdx), k);

    return indices;
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    numClasses = size(targets, 2);
    N = size(targets, 1);

    indices = Array{Int64,1}(undef, N);

    for c in 1:numClasses
        @assert sum(targets[:,c]) >= k "La clase $c no tiene suficientes patrones"
        indices[targets[:,c]] = crossvalidation(sum(targets[:,c]), k);
    end;

    return indices;
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    oneHotTargets = oneHotEncoding(targets);
    return crossvalidation(oneHotTargets, k);
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    #
    # Codigo a desarrollar
    #
end;
