using Random
using Statistics
using DelimitedFiles

Random.seed!(1234);

include("firmas.jl")
include("extractFeatures.jl")

# Función principal
function ejecutarModeloANN(inputs, targets, crossValidationIndices)
    println("=== Modelo 1: Redes Neuronales Artificiales ===");
    println();
    
    # 8 topologías a probar 
    topologies = [
        [32],           # 1 capa oculta con 32 neuronas
        [64],           # 1 capa oculta con 64 neuronas
        [128],          # 1 capa oculta con 128 neuronas
        [256],          # 1 capa oculta con 256 neuronas
        [64, 32],       # 2 capas ocultas (64→32)
        [128, 64],      # 2 capas ocultas (128→64)
        [256, 128],     # 2 capas ocultas (256→128)
        [128, 128]      # 2 capas ocultas (128→128)
    ]
    
    # Parametros comuns para entrenar (Ter solo 1 descomentado)
    learning_rate = 0.01; num_executions = 5; max_epochs = 500; min_loss = 0.1;
    #learning_rate = 0.005; num_executions = 5; max_epochs = 500; min_loss = 0.1;
    #learning_rate = 0.05; num_executions = 5; max_epochs = 500; min_loss = 0.05;
    
    # Almacenar resultados
    results = [];
    
    # Probar cada topologia
    for topology in topologies
        println("========== Evaluando topologia: ", topology , " ===========");
        
        try
            # Crear diccionario de hiperparametros para topología
            hyperparameters = Dict(
                "topology" => topology,
                "numExecutions" => num_executions,
                "maxEpochs" => max_epochs,
                "minLoss" => min_loss,
                "learningRate" => learning_rate
            );
            
            # Validacion cruzada
            accuracy, error_rate, recall, specificity, precision, npv, f1, conf_matrix = 
                modelCrossValidation(
                    :ANN,                   # Tipo de modelo
                    hyperparameters,        # Hiperparametros
                    (inputs, targets),      # Dataset
                    crossValidationIndices  # Índices de validación cruzada
                );

            println("Matriz de confusión de test:");
            display(conf_matrix);
            println();

            # Gardar resultados
            push!(results, (
                topology=topology,
                accuracy=accuracy,
                error_rate=error_rate,
                recall=recall,
                specificity=specificity,
                precision=precision,
                npv=npv,
                f1=f1,
                conf_matrix=conf_matrix
            ))
            
            # Mostrar resultados
            println("  Accuracy: ", accuracy[1], " ± ", accuracy[2]);
            println("  Sensibilidad (Recall): ", recall[1], " ± ", recall[2]);
            println("  Especificidad: ", specificity[1], " ± ", specificity[2]);
            println("  VPP (Precision): ", precision[1], " ± ", precision[2]);
            println("  F1-Score: ", f1[1], " ± ", f1[2]);
            println();
        catch e
            println("  Error al evaluar topología ", topology, ": ", e);
            println("  Saltando a la siguiente topología...");
            println();
        end
    end
    
    # Encontrar mellor topologia basada en F1-score
    best_idx = argmax([r.f1[1] for r in results]);
    best_topology = results[best_idx].topology;
    best_f1 = results[best_idx].f1;
    best_conf_matrix = results[best_idx].conf_matrix;
    
    # Entrenar con TODOS los datos para obtener métricas de training
    println("\n========= Entrenando modelo final con todos los datos =========");
    hyperparameters_final = Dict(
        "topology" => best_topology,
        "numExecutions" => 1,  # Sin repeticiones para training
        "maxEpochs" => max_epochs,
        "minLoss" => min_loss,
        "learningRate" => learning_rate
    );
    
    # Crear índices de train (todos) y test (vacío) para obtener métricas de training
    all_indices = collect(1:size(inputs, 1));
    train_metrics = modelCrossValidation(
        :ANN,
        hyperparameters_final,
        (inputs, targets),
        all_indices  # Todos los datos juntos
    );
    
    train_accuracy = train_metrics[1];
    train_recall = train_metrics[3];
    train_specificity = train_metrics[4];
    train_precision = train_metrics[5];
    train_f1 = train_metrics[7];
    train_conf_matrix = train_metrics[8];
    
    println("========= Resultados Finales =========");
    println("Mejor topologia (por F1-score): ", best_topology);
    println("\n--- VALIDACIÓN CRUZADA (Test) ---");
    println("Accuracy: ", results[best_idx].accuracy[1], " ± ", results[best_idx].accuracy[2]);
    println("Sensibilidad (Recall): ", results[best_idx].recall[1], " ± ", results[best_idx].recall[2]);
    println("Especificidad: ", results[best_idx].specificity[1], " ± ", results[best_idx].specificity[2]);
    println("VPP (Precision): ", results[best_idx].precision[1], " ± ", results[best_idx].precision[2]);
    println("F1-Score: ", best_f1[1], " ± ", best_f1[2]);
    println("\n--- TRAINING (Todo el dataset) ---");
    println("Accuracy: ", train_accuracy[1]);
    println("Sensibilidad (Recall): ", train_recall[1]);
    println("Especificidad: ", train_specificity[1]);
    println("VPP (Precision): ", train_precision[1]);
    println("F1-Score: ", train_f1[1]);
    println();
    println("Matriz de confusión (mejor topología - validación cruzada):");
    display(best_conf_matrix);
    println();
    
    return results, best_idx, best_conf_matrix, train_accuracy, train_recall, train_specificity, train_precision, train_f1, train_conf_matrix;
end

# Programa principal
# Verifica si datos ya están cargados (por script_resultados.jl)
# Si no, los carga de forma independiente
if !(@isdefined inputs) || !(@isdefined targets) || !(@isdefined cvIndices)
    println("Cargando datos...");
    inputs, targets = loadDataset("./dataset");

    println("Generando índices de validación cruzada...");
    cvIndices = crossvalidation(targets, 10);
end

println("Ejecutando experimentos con Redes Neuronales...");
println();

results, best_idx, best_conf_matrix, train_accuracy, train_recall, train_specificity, train_precision, train_f1, train_conf_matrix = ejecutarModeloANN(inputs, targets, cvIndices);
