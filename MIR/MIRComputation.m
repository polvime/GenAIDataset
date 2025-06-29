% Full MIR Analysis Script (Corrected & Final)
% Folder path for your audio files
mirwaitbar(0);

folderPath = 'Soundraw';  % Your audio folder
audioFiles = dir(fullfile(folderPath, '*.mp3'));

% Initialize storage
fileNames = {};
rmsList = []; lowEnergyList = []; centroidList = []; spreadList = [];
fluxList = []; roughnessList = []; brightnessList = []; mfccMeanList = [];
modeList = {}; modeStrengthList = []; keyNameList = {}; keyClarityList = []; hcdfList = [];
eventDensityList = []; pulseClarityList = []; tempoList = [];
fluctuationCentroidList = []; fluctuationEntropyList = [];
attackTimeList = []; attackSlopeList = []; noveltyList = [];

% Analysis loop
for i = 1:length(audioFiles)
    filePath = fullfile(folderPath, audioFiles(i).name);
    fileNames{end+1} = audioFiles(i).name;
    fprintf('🎧 Processing: %s\n', audioFiles(i).name);

    try
        % Dynamics
        rms = mean(mirgetdata(mirrms(filePath)));
        lowEnergy = mean(mirgetdata(mirlowenergy(filePath)));

        % Timbre
        centroid = mean(mirgetdata(mircentroid(filePath)));
        spread = mean(mirgetdata(mirspread(filePath)));
        flux = mean(mirgetdata(mirflux(filePath)));
        roughness = mean(mirgetdata(mirroughness(filePath)));
        brightness = mean(mirgetdata(mirbrightness(filePath)));
        mfcc = mirgetdata(mirmfcc(filePath));
        mfccMean = mean(mfcc(:));

        % Harmony
        mode = mirgetdata(mirmode(filePath));
        [keyName, keyClarity] = mirgetdata(mirkey(filePath));
        hcdf = mean(mirgetdata(mirhcdf(filePath)));

        % Rhythm
        eventDensity = mean(mirgetdata(mireventdensity(filePath)));
        pulseClarity = mean(mirgetdata(mirpulseclarity(filePath)));
        tempo = mean(mirgetdata(mirtempo(filePath)));
        fluctuationCentroid = mean(mirgetdata(mircentroid(filePath)));
        fluctuationEntropy = mean(mirgetdata(mirentropy(filePath)));

        % Articulation
        attackTime = mean(mirgetdata(mirattacktime(filePath)));
        attackSlope = mean(mirgetdata(mirattackslope(filePath)), 'omitnan');

        % Structure
        novelty = mean(mirgetdata(mirnovelty(filePath)));

        % Safe assignment
        rmsList(end+1) = safeVal(rms);
        lowEnergyList(end+1) = safeVal(lowEnergy);
        centroidList(end+1) = safeVal(centroid);
        spreadList(end+1) = safeVal(spread);
        fluxList(end+1) = safeVal(flux);
        roughnessList(end+1) = safeVal(roughness);
        brightnessList(end+1) = safeVal(brightness);
        mfccMeanList(end+1) = safeVal(mfccMean);
        
        if isnan(mode)
            keyModeLabel = "Unknown";
        elseif mode > 0
            keyModeLabel = "Major";
        elseif mode < 0
            keyModeLabel = "Minor";
        else
            keyModeLabel = "Ambiguous";
        end
        
        modeList{end+1} = keyModeLabel;
        modeStrengthList(end+1) = abs(mode);  % strength as absolute value

        try
            keyNameList{end+1} = keyName;
        catch
            keyNameList{end+1} = "Unknown";
        end

        keyClarityList(end+1) = safeVal(keyClarity);
        hcdfList(end+1) = safeVal(hcdf);
        eventDensityList(end+1) = safeVal(eventDensity);
        pulseClarityList(end+1) = safeVal(pulseClarity);
        tempoList(end+1) = safeVal(tempo);
        fluctuationCentroidList(end+1) = safeVal(fluctuationCentroid);
        fluctuationEntropyList(end+1) = safeVal(fluctuationEntropy);
        attackTimeList(end+1) = safeVal(attackTime);
        attackSlopeList(end+1) = safeVal(attackSlope);
        noveltyList(end+1) = safeVal(novelty);

        fprintf('✅ Success: %s\n', audioFiles(i).name);

    catch ME
        % Log error
        fprintf('❌ Error in file %s: %s\n', audioFiles(i).name, ME.message);

        % Push fallback data
        rmsList(end+1) = NaN; lowEnergyList(end+1) = NaN; centroidList(end+1) = NaN;
        spreadList(end+1) = NaN; fluxList(end+1) = NaN; roughnessList(end+1) = NaN;
        brightnessList(end+1) = NaN; mfccMeanList(end+1) = NaN;
        modeList{end+1} = "Error"; modeStrengthList{end+1} = NaN; keyNameList{end+1} = "Error"; keyClarityList(end+1) = NaN;
        hcdfList(end+1) = NaN; eventDensityList(end+1) = NaN; pulseClarityList(end+1) = NaN;
        tempoList(end+1) = NaN; fluctuationCentroidList(end+1) = NaN;
        fluctuationEntropyList(end+1) = NaN; attackTimeList(end+1) = NaN;
        attackSlopeList(end+1) = NaN; noveltyList(end+1) = NaN;
    end
end

modeList = modeList';
modeStrengthList = modeStrengthList(:);  % Convert to column if numeric
keyNameList = keyNameList';

vars = {
    fileNames, rmsList, lowEnergyList, centroidList, spreadList, fluxList, roughnessList, ...
    brightnessList, mfccMeanList, modeList, modeStrengthList, keyNameList, keyClarityList, ...
    hcdfList, eventDensityList, pulseClarityList, tempoList, fluctuationCentroidList, ...
    fluctuationEntropyList, attackTimeList, attackSlopeList, noveltyList};

% Check lengths
for v = 1:length(vars)
    fprintf('%d: Length = %d\n', v, length(vars{v}));
end

% Build table
resultsTable = table(fileNames(:), rmsList(:), lowEnergyList(:), centroidList(:), spreadList(:), ...
    fluxList(:), roughnessList(:), brightnessList(:), mfccMeanList(:), modeList(:), ...
    modeStrengthList(:), keyNameList(:), keyClarityList(:), hcdfList(:), eventDensityList(:), ...
    pulseClarityList(:), tempoList(:), fluctuationCentroidList(:), fluctuationEntropyList(:), ...
    attackTimeList(:), attackSlopeList(:), noveltyList(:), ...
    'VariableNames', {'FileName', 'RMS', 'LowEnergy', 'SpectralCentroid', 'SpectralSpread', ...
    'SpectralFlux', 'Roughness', 'Brightness', 'MFCC_Mean', 'Mode', 'ModeStrength', ...
    'KeyName', 'KeyClarity', 'HCDF', 'EventDensity', 'PulseClarity', 'Tempo', ...
    'FluctuationCentroid', 'FluctuationEntropy', 'AttackTime', 'AttackSlope', 'Novelty'});

% Save to CSV
outputFile = fullfile(folderPath, 'MIR_Results_Final.csv');
writetable(resultsTable, outputFile);

fprintf('\n📁 All Done! Results saved to: %s\n', outputFile);

% === Helper function ===
function val = safeVal(input)
    if isempty(input) || ~isnumeric(input)
        val = NaN;
    else
        val = input;
    end
end