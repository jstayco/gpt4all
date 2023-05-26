#include <mutex>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QSettings>

#include "aimodels.h"
#include "download.h"
#include "chat.h"

AIModels::AIModels()
{
    m_regexGGML = QRegularExpression("ggml-");
    m_regexBinSuffix = QRegularExpression("\\.bin$");
    m_regexWordStart = QRegularExpression("\\b\\w");
    m_regexDigitB = QRegularExpression("(\\d)B", QRegularExpression::CaseInsensitiveOption);
    m_regexGPT4All = QRegularExpression("gpt4all", QRegularExpression::CaseInsensitiveOption);
    m_regexGPT = QRegularExpression("gpt", QRegularExpression::CaseInsensitiveOption);
    m_regexDoubleGPT = QRegularExpression("gpt gpt", QRegularExpression::CaseInsensitiveOption);
    m_regexQuantization = QRegularExpression("(q)(\\d+)(_?(\\d+)?)", QRegularExpression::CaseInsensitiveOption);
    connect(Download::globalInstance(), &Download::downloadFinished, this, &AIModels::invalidateModelListCache);
    connect(Download::globalInstance(), &Download::modelListChanged, this, &AIModels::invalidateModelListCache);
}

void AIModels::addModel(const QString& modelName,
                         const QString& modelDisplayName,
                         const QString& source,
                         bool isInstalled,
                         const QVariantMap& optionalProperties) {
    ModelData data;
    data.modelName = modelName;
    data.modelDisplayName = modelDisplayName;
    data.source = source;
    data.isInstalled = isInstalled;
    data.optionalProperties = optionalProperties;

    m_models[modelName] = data;

    emit modelListChanged();
}

QVariantMap AIModels::getModel(const QString& modelName) const {
    if (m_models.contains(modelName)) {
        const ModelData& data = m_models[modelName];
        QVariantMap map;
        map["modelName"] = data.modelName;
        map["modelDisplayName"] = data.modelDisplayName;
        map["source"] = data.source;
        map["isInstalled"] = data.isInstalled;
        for(auto key : data.optionalProperties.keys()){
            map[key] = data.optionalProperties[key];
        }
        return map;
    }
    return QVariantMap();
}

QVariantMap AIModels::getModel(int index) const {
    if (index >= 0 && index < m_models.size()) {
        QString key = m_models.keys().at(index);
        return getModel(key);
    }
    return QVariantMap();
}

QString AIModels::getModelName(const QString& modelName) const {
    QVariantMap model = getModel(modelName);
    if (!model.isEmpty()) {
        return model["modelName"].toString();
    }
    return QString();
}

void AIModels::handleModelNameChanged(const QString& modelName) {
    m_currentModelName = modelName;
}

void AIModels::updateCurrentModelName(const QString& modelName) {
    m_currentModelName = modelName;
}

int AIModels::size() const {
    return m_models.size();
}

QList<QVariantMap> AIModels::modelList() const
{
    // Build a model list from exepath and from the localpath
    static QList<QVariantMap> list;

    // lock mutex before checking or modifying 'initialized' and 'list'
    std::lock_guard<std::mutex> guard(m_modelListMutex);

    if (m_isModelListInitialized) {
        return list;
    }

    QString exePath = QCoreApplication::applicationDirPath() + QDir::separator();
    QString localPath = Download::globalInstance()->downloadLocalModelsPath();

    QSettings settings;
    settings.sync();
    // The user default model can be set by the user in the settings dialog. The "default" user
    // default model is "Application default" which signals we should use the default model that was
    // specified by the models.json file.
    QString defaultModel = settings.value("userDefaultModel").toString();
    if (defaultModel.isEmpty() || defaultModel == "Application default")
        defaultModel = settings.value("defaultModel").toString();

    QString currentModelName = m_currentModelName.isEmpty() ? defaultModel : m_currentModelName;

    {
        QDir dir(exePath);
        dir.setNameFilters(QStringList() << "ggml-*.bin");
        QStringList fileNames = dir.entryList();
        for (QString f : fileNames) {
            QString filePath = exePath + f;
            QFileInfo info(filePath);
            QString name = info.completeBaseName().remove(0, 5);
            if (info.exists()) {
                QVariantMap model;
                model["original"] = name;
                model["formatted"] = formatModelName(name, false);
                if (name == currentModelName)
                    list.prepend(model);
                else
                    list.append(model);
            }
        }
    }

    if (localPath != exePath) {
        QDir dir(localPath);
        dir.setNameFilters(QStringList() << "ggml-*.bin" << "chatgpt-*.txt");
        QStringList fileNames = dir.entryList();
        for (QString f : fileNames) {
            QString filePath = localPath + f;
            QFileInfo info(filePath);
            QString basename = info.completeBaseName();
            QString name = basename.startsWith("ggml-") ? basename.remove(0, 5) : basename;
            if (info.exists() && !listContainsOriginalName(list, name)) { // don't allow duplicates
                QVariantMap model;
                model["original"] = name;
                model["formatted"] = formatModelName(name, basename.startsWith("chatgpt-"));
                if (name == currentModelName)
                    list.prepend(model);
                else
                    list.append(model);
            }
        }
    }


    if (list.isEmpty()) {
        if (exePath != localPath) {
            qWarning() << "ERROR: Could not find any applicable models in"
                       << exePath << "nor" << localPath;
        } else {
            qWarning() << "ERROR: Could not find any applicable models in"
                       << exePath;
        }
        return QList<QVariantMap>();
    }

    m_isModelListInitialized = true;
    return list;
}

void AIModels::invalidateModelListCache()
{
    // lock mutex before modifying 'initialized'
    std::lock_guard<std::mutex> guard(m_modelListMutex);

    m_isModelListInitialized = false;
}

bool AIModels::listContainsOriginalName(QList<QVariantMap> list, QString name) const
{
    for (QVariantMap model : list) {
        if (model["original"].toString() == name) {
            return true;
        }
    }
    return false;
}

QString AIModels::formatModelName(QString filename, bool isChatGPT) const
{
    QString name = filename;
    if (!isChatGPT) {
        // Remove prefixes and sufixes
        name.remove(m_regexGGML);
        name.remove(m_regexBinSuffix);

        // Replace delims with spaces
        name.replace('-', ' ');
        name = name.replace('.', ' ');

        // Normalize all words to lower to make editing easier
        name = name.toLower();

        // Uppercase every word for nice Title Case.
        QRegularExpressionMatch match = m_regexWordStart.match(name);
        while (match.hasMatch()) {
            name.replace(match.capturedStart(), 1, match.captured().toUpper());
            match = m_regexWordStart.match(name, match.capturedEnd());
        }

    }

    // Normalize GPT stylings
    name.replace(m_regexGPT4All, "GPT4All");
    name.replace(m_regexGPT, "GPT");
    name.replace(m_regexDoubleGPT, "GPT");

    // Capitalize the b in models (13b to 13B)
    QRegularExpressionMatch match = m_regexDigitB.match(name);
    while (match.hasMatch()) {
        name.replace(match.capturedStart(), match.capturedLength(), match.captured(1) + 'B');
        match = m_regexDigitB.match(name, match.capturedEnd());
    }

    // Turns quantization measurements to nice readable strings for users
    // Q4_2 = 4.2q. Q4_0 and Q4 = 4q.
    QRegularExpressionMatch matchQuantization = m_regexQuantization.match(name);
    while (matchQuantization.hasMatch()) {
        QString replacement;
        if (!matchQuantization.captured(3).isEmpty() && matchQuantization.captured(3) != "0") {
            replacement = matchQuantization.captured(2) + "." + matchQuantization.captured(4) + "q";
        }
        else {
            replacement = matchQuantization.captured(2) + "q";
        }
        name.replace(matchQuantization.capturedStart(), matchQuantization.capturedLength(), replacement);
        matchQuantization = m_regexQuantization.match(name, matchQuantization.capturedStart() + replacement.length());
    }

    return name;
}
