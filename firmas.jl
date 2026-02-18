
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
### Esta funcion nn tou seuguro de que te ben, pide non usar function e esto é o unico que lle vexo sentido e non me salta errores xd, despos proboa ###

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
    datasets = copy(dataset);
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
    # Si tiene 1 columna: aplica threshold
    # Si tiene >1 columna: selecciona el máximo de cada fila
    if size(outputs, 2) == 1;
        return outputs .>= threshold;
    else;
        classifiedOutputs = falses(size(outputs));
        for i in 1:size(outputs, 1);
            classifiedOutputs[i, argmax(outputs[i, :])] = true;
        end;
        return classifiedOutputs;
    end;
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
    else        ### Ahora funciona ben con 2 ou menos. i ao salir do bucle tiña leght(topology) + 1 como valor e transferFunctions[] solo ten length(top..) elementos, de ahi o BoundsError, intentaba acceder ao 4 cando solo tiña 3 elementos ###
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    end
    return ann;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    inputs = Float32.(dataset[1]);
    targets = dataset[2];
    ann = buildClassANN(size(inputs, 2), topology, size(targets, 2); transferFunctions);

    ### A funcion "loss" é pa decidir que funcion usar para entrenar a RNA, recibe as salidas do modelo e as salidas deseadas ###
    if (size(targets, 2) == 1);
        loss(x,y) = Losses.binarycrossentropy(ann(x),y);
    else;
        loss(x,y) = Losses.crossentropy(ann(x),y);
    end;

    ###Este é o vector cos valores de loss en cada ciclo ###
    lossValues = zeros(Float32, maxEpochs+1);
    lossValues[1] = loss(inputs', targets');

    currentEpoch = 1;
    while (currentEpoch <= maxEpochs) && (lossValues[currentEpoch] > minLoss);
        Flux.train!(loss, Flux.params(ann), [(inputs', targets')], ADAM(learningRate));
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
    #
    # Codigo a desarrollar
    #
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
    if (size(targets, 2) == 1);
        loss(x,y) = Losses.binarycrossentropy(ann(x),y);
    else;
        loss(x,y) = Losses.crossentropy(ann(x),y);
    end;

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

    while currentEpoch <= maxEpochs
        # Training de un ciclo
        Flux.train!(loss, Flux.params(ann), [(inputs', targets')], ADAM(learningRate));

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
    #
    # Codigo a desarrollar
    #
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
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
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
