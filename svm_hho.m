% Veri içe aktarma
df = readtable('Breast_cancer2.csv');

% Veri ön işleme
X = table2array(df(:, 3:32));
Y = table2array(df(:, 2));

% Veriyi eğitim ve test setlerine ayırma
cv = cvpartition(Y, 'holdout', 0.3);
X_Train = X(training(cv, 1), :);
y_Train = Y(training(cv, 1));
X_Test = X(test(cv, 1), :);
y_Test = Y(test(cv, 1));

% Eğitim verisini normalize etme ve normalizasyon parametrelerini kaydetme
[X_Train, mu, stddev] = normalize(X_Train);

% Test verisini eğitim verisinin normalizasyon parametreleriyle normalize etme
for i = 1:size(X_Test, 2)
    X_Test(:, i) = (X_Test(:, i) - mu(1, i)) / stddev(1, i);
end

% Harris Hawk Optimizasyonu (HHO) parametreleri
N = 30;
T = 50;

% HHO algoritması için sınırlar
lb = [-3, -3];
ub = [10, 10];
dim = numel(lb);

% HHO ile SVM modelini eğitme ve performans metriklerini elde etme
[Rabbit_Energy, Rabbit_Location, CNVG, BestFitness, AccuracyList, F1List] = HHO_with_metrics(N, T, lb, ub, dim, @svm_fitness, X_Train, y_Train, X_Test, y_Test);

% En iyi fitness değerlerini gösterme
disp('Her iterasyondaki en iyi fitness değerleri:');
disp(BestFitness);

% Her iterasyondaki accuracy, F1 ve fitness değerlerini gösterme
disp('Her iterasyondaki Accuracy, F1 ve Fitness değerleri:');
for iter = 1:T
    disp(['Iterasyon ' num2str(iter) ', Fitness: ' num2str(BestFitness(iter)) ', Accuracy: ' num2str(AccuracyList(iter)) ', F1 Score: ' num2str(F1List(iter))]);
end

% En sonunda elde edilen en iyi fitness değeri
finalBestFitness = min(BestFitness);
disp(['En sonunda elde edilen en iyi fitness değeri: ', num2str(finalBestFitness)]);

% Ortalama accuracy, F1 ve fitness değerlerini hesapla ve göster
avgAccuracy = mean(AccuracyList);
avgF1 = mean(F1List);
disp(['Ortalama Accuracy: ' num2str(avgAccuracy) ', Ortalama F1 Score: ' num2str(avgF1) ', Ortalama Fitness: ' num2str(finalBestFitness)]);

% Harris Hawk Optimizasyonu: Harris Hawk Optimizasyonu algoritması ile SVM modelini eğitme
function [Rabbit_Energy, Rabbit_Location, CNVG, BestFitness, AccuracyList, F1List] = HHO_with_metrics(N, T, lb, ub, dim, fobj, X_Train, y_Train, X_Test, y_Test)
    disp('HHO is now tackling your problem');
    tic

    % Başlangıçta tavşanın enerjisi sonsuz kabul edilir
    Rabbit_Location = zeros(1, dim);
    Rabbit_Energy = inf;

    % Harris'ın şahinlerinin başlangıç konumları
    X = initialization(N, dim, ub, lb);

    % Değişkenler
    CNVG = zeros(1, T);
    BestFitness = zeros(1, T);
    AccuracyList = zeros(1, T);
    F1List = zeros(1, T);

    t = 0; % Döngü sayacı

    % Ana döngü
    while t < T
        % Her bir şahinin konumunu güncelle
        for i = 1:size(X, 1)
            % Sınırları kontrol et
            FU = X(i, :) > ub;
            FL = X(i, :) < lb;
            X(i, :) = X(i, :) .* (~(FU + FL)) + ub .* FU + lb .* FL;

            % Konumun fitness değerini hesapla
            [fitness, yPred] = fobj(X(i, :), X_Train, y_Train, X_Test, y_Test);

            % Eğer bulunan fitness, tavşanınkinin altındaysa, tavşanın yerini ve enerjisini güncelle
            if fitness < Rabbit_Energy
                Rabbit_Energy = fitness;
                Rabbit_Location = X(i, :);
            end
        end

        % Tavşanın enerjisini güncelleme
        E1 = 2 * (1 - (t / T));

        % Her bir şahinin konumunu güncelleme
        for i = 1:size(X, 1)
            % Rastgele bir enerji değeri
            E0 = 2 * rand() - 1;
            Escaping_Energy = E1 * E0;

            % Eğer kaçış enerjisi 1'den büyükse keşif yap
            if abs(Escaping_Energy) >= 1
                % Keşif
                q = rand();
                rand_Hawk_index = floor(N * rand() + 1);
                X_rand = X(rand_Hawk_index, :);

                % Rastgele bir strateji seç
                if q < 0.5
                    % Diğer aile üyelerine dayanarak konumunu belirle
                    X(i, :) = X_rand - rand() * abs(X_rand - 2 * rand() * X(i, :));
                elseif q >= 0.5
                    % Rastgele bir yüksek ağacın üzerine konum belirle
                    X(i, :) = (Rabbit_Location(1, :) - mean(X)) - rand() * ((ub - lb) .* rand + lb);
                end
            % Eğer kaçış enerjisi 1'den küçükse, sömürü yap
            elseif abs(Escaping_Energy) < 1
                % Sömürü
                q = rand();
                rand_Hawk_index = floor(N * rand() + 1);
                X_rand = X(rand_Hawk_index, :);

                % Rastgele bir strateji seç
                if q < 0.5
                    % Sert kuşatma stratejisi
                    Jump_strength = 2 * (1 - rand());
                    X(i, :) = (Rabbit_Location) - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :));
                elseif q >= 0.5
                    % Yumuşak kuşatma stratejisi
                    Jump_strength = 2 * (1 - rand());
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :));
                    % İyileştirilmiş bir hamle mi?
                    if fobj(X1, X_Train, y_Train, X_Test, y_Test) < fobj(X(i, :), X_Train, y_Train, X_Test, y_Test)
                        X(i, :) = X1;
                    else
                        % Yeniden iyileştirme stratejisi
                        Jump_strength = 2 * (1 - rand());
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :)) + rand(1, dim) .* Levy(dim);
                        % İyileştirilmiş bir hamle mi?
                        if fobj(X2, X_Train, y_Train, X_Test, y_Test) < fobj(X(i, :), X_Train, y_Train, X_Test, y_Test)
                            X(i, :) = X2;
                        end
                    end
                end
            end

            % Konumun fitness değerini hesapla
            [fitness, yPred] = fobj(X(i, :), X_Train, y_Train, X_Test, y_Test);

            % Eğer bulunan fitness, tavşanınkinin altındaysa, tavşanın yerini ve enerjisini güncelle
            if fitness < Rabbit_Energy
                Rabbit_Energy = fitness;
                Rabbit_Location = X(i, :);
            end
        end

        % Sonuçları kaydet
        CNVG(t + 1) = Rabbit_Energy;
        BestFitness(t + 1) = Rabbit_Energy;

        % Doğruluk ve F1 skorunu hesapla
        [accuracy, f1] = evaluate_classification(yPred, y_Test);
        AccuracyList(t + 1) = accuracy;
        
        % F1List vektörünün boyutunu güncelle ve yeni değeri ekle
        F1List = [F1List(1:t) f1 zeros(1, T - t)];
        
        disp(['Iterasyon ' num2str(t) ', En İyi Fitness: ' num2str(Rabbit_Energy) ', Accuracy: ' num2str(accuracy) ', F1 Skoru: ' num2str(f1)]);

        t = t + 1;
    end

    % Zamanı ölç ve ekrana yazdır
    toc
end

% SVM Fitness Fonksiyonu: SVM modelini eğitip fitness değerini hesapla
function [fitness, yPred] = svm_fitness(x, X_Train, y_Train, X_Test, y_Test)
    % Optimizasyon vektöründen kutu kısıtlaması ve çekirdek ölçeği değerlerini çıkart
    boxConstraint = 10^x(1);
    kernelScale = 10^x(2);

    % Kutu kısıtlamasını makul bir aralıkta olacak şekilde kontrol et
    boxConstraint = max(boxConstraint, 1e-3);

    try
        % SVM modelini eğit
        svmModel = fitcsvm(X_Train, y_Train, 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint, 'KernelScale', kernelScale);

        % SVM modelini test et
        [yPred, ~] = predict(svmModel, X_Test);

        % Hata oranını hesapla (ne kadar düşükse, fitness o kadar iyi)
        err_svm = sum(yPred ~= y_Test) / numel(y_Test);

        % Fitness değeri
        fitness = err_svm;
    catch
        % Eğer SVM eğitiminde bir hata oluşursa (örneğin, çok küçük bir kutu kısıtlamasından dolayı),
        % bu tür çözümleri caydırmak için büyük bir fitness değeri ata
        fitness = 1e6;
    end
end

% Sınıflandırma Performansını Değerlendir: Doğruluk ve F1 skorunu hesapla
function [accuracy, f1] = evaluate_classification(yPred, yTrue)
    cm = confusionmat(yTrue, yPred);
    % Doğruluğu hesapla
    accuracy = sum(diag(cm)) / sum(cm, 'all');
    % Precision, recall ve F1 skorunu hesapla
    precision = diag(cm) ./ sum(cm, 1)';
    recall = diag(cm) ./ sum(cm, 2);
    f1 = 2 * (precision .* recall) / (precision + recall);
    f1(isnan(f1)) = 0;  % NaN değerlerini sıfıra ayarla
    f1 = mean(f1);
end

% Levy Uçuşu: Levy uçuşunu gerçekleştir
function Levy_step = Levy(dim)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
    u = 0.01 * randn(1, dim) * sigma;
    v = randn(1, dim);
    zz = abs(v) .^ (1 / beta);
    Levy_step = u ./ zz;
end

% Başlangıç Popülasyonunu Oluştur: Rastgele başlangıç konumları oluştur
function X = initialization(N, dim, ub, lb)
    X = zeros(N, dim);
    for i = 1:N
        X(i, :) = lb + (ub - lb) .* rand(1, dim);
    end
end
