#include "chat.h"
#include "llm.h"
#include "network.h"
#include "download.h"

Chat::Chat(QObject *parent)
    : QObject(parent)
    , m_id(Network::globalInstance()->generateUniqueId())
    , m_name(tr("New Chat"))
    , m_chatModel(new ChatModel(this))
    , m_responseInProgress(false)
    , m_creationDate(QDateTime::currentSecsSinceEpoch())
    , m_llmodel(new ChatLLM(this))
    , m_isServer(false)
    , m_shouldDeleteLater(false)
{
    connectLLM();
}

// Initialize our static properties
const QRegularExpression Chat::m_regexGGML("ggml-");
const QRegularExpression Chat::m_regexBinSuffix("(\\.q\\d+(_\\d+)?\\.bin|\\.bin)$");
const QRegularExpression Chat::m_regexWordStart("\\b\\w");
const QRegularExpression Chat::m_regexDigitB("(\\d{1,2})b");
const QRegularExpression Chat::m_regexGPT4All("gpt4all", QRegularExpression::CaseInsensitiveOption);
const QRegularExpression Chat::m_regexGPT("gpt(?!4all)", QRegularExpression::CaseInsensitiveOption);
const QRegularExpression Chat::m_regexDoubleGPT("(GPT)\\s\\1", QRegularExpression::CaseInsensitiveOption);
const QRegularExpression Chat::m_regexQuantization("(\\.?)q(\\d+)(_\\d+)?", QRegularExpression::CaseInsensitiveOption);

Chat::Chat(bool isServer, QObject *parent)
    : QObject(parent)
    , m_id(Network::globalInstance()->generateUniqueId())
    , m_name(tr("Server Chat"))
    , m_chatModel(new ChatModel(this))
    , m_responseInProgress(false)
    , m_creationDate(QDateTime::currentSecsSinceEpoch())
    , m_llmodel(new Server(this))
    , m_isServer(true)
    , m_shouldDeleteLater(false)
{
    connectLLM();
}

Chat::~Chat()
{
    delete m_llmodel;
    m_llmodel = nullptr;
}

void Chat::connectLLM()
{
    // Should be in same thread
    connect(Download::globalInstance(), &Download::modelListChanged, this, &Chat::modelListChanged, Qt::DirectConnection);
    connect(this, &Chat::modelNameChanged, this, &Chat::modelListChanged, Qt::DirectConnection);

    // Should be in different threads
    connect(m_llmodel, &ChatLLM::isModelLoadedChanged, this, &Chat::isModelLoadedChanged, Qt::QueuedConnection);
    connect(m_llmodel, &ChatLLM::isModelLoadedChanged, this, &Chat::handleModelLoadedChanged, Qt::QueuedConnection);
    connect(m_llmodel, &ChatLLM::responseChanged, this, &Chat::handleResponseChanged, Qt::QueuedConnection);
    connect(m_llmodel, &ChatLLM::responseStarted, this, &Chat::responseStarted, Qt::QueuedConnection);
    connect(m_llmodel, &ChatLLM::responseStopped, this, &Chat::responseStopped, Qt::QueuedConnection);
    connect(m_llmodel, &ChatLLM::modelNameChanged, this, &Chat::handleModelNameChanged, Qt::QueuedConnection);
    connect(m_llmodel, &ChatLLM::modelLoadingError, this, &Chat::modelLoadingError, Qt::QueuedConnection);
    connect(m_llmodel, &ChatLLM::recalcChanged, this, &Chat::handleRecalculating, Qt::QueuedConnection);
    connect(m_llmodel, &ChatLLM::generatedNameChanged, this, &Chat::generatedNameChanged, Qt::QueuedConnection);

    connect(this, &Chat::promptRequested, m_llmodel, &ChatLLM::prompt, Qt::QueuedConnection);
    connect(this, &Chat::modelNameChangeRequested, m_llmodel, &ChatLLM::modelNameChangeRequested, Qt::QueuedConnection);
    connect(this, &Chat::loadDefaultModelRequested, m_llmodel, &ChatLLM::loadDefaultModel, Qt::QueuedConnection);
    connect(this, &Chat::loadModelRequested, m_llmodel, &ChatLLM::loadModel, Qt::QueuedConnection);
    connect(this, &Chat::generateNameRequested, m_llmodel, &ChatLLM::generateName, Qt::QueuedConnection);

    // The following are blocking operations and will block the gui thread, therefore must be fast
    // to respond to
    connect(this, &Chat::regenerateResponseRequested, m_llmodel, &ChatLLM::regenerateResponse, Qt::BlockingQueuedConnection);
    connect(this, &Chat::resetResponseRequested, m_llmodel, &ChatLLM::resetResponse, Qt::BlockingQueuedConnection);
    connect(this, &Chat::resetContextRequested, m_llmodel, &ChatLLM::resetContext, Qt::BlockingQueuedConnection);
}

void Chat::reset()
{
    stopGenerating();
    // Erase our current on disk representation as we're completely resetting the chat along with id
    LLM::globalInstance()->chatListModel()->removeChatFile(this);
    emit resetContextRequested(); // blocking queued connection
    m_id = Network::globalInstance()->generateUniqueId();
    emit idChanged();
    // NOTE: We deliberately do no reset the name or creation date to indictate that this was originally
    // an older chat that was reset for another purpose. Resetting this data will lead to the chat
    // name label changing back to 'New Chat' and showing up in the chat model list as a 'New Chat'
    // further down in the list. This might surprise the user. In the future, we me might get rid of
    // the "reset context" button in the UI. Right now, by changing the model in the combobox dropdown
    // we effectively do a reset context. We *have* to do this right now when switching between different
    // types of models. The only way to get rid of that would be a very long recalculate where we rebuild
    // the context if we switch between different types of models. Probably the right way to fix this
    // is to allow switching models but throwing up a dialog warning users if we switch between types
    // of models that a long recalculation will ensue.
    m_chatModel->clear();
}

bool Chat::isModelLoaded() const
{
    return m_llmodel->isModelLoaded();
}

void Chat::prompt(const QString &prompt, const QString &prompt_template, int32_t n_predict,
    int32_t top_k, float top_p, float temp, int32_t n_batch, float repeat_penalty,
    int32_t repeat_penalty_tokens)
{
    emit promptRequested(prompt, prompt_template, n_predict, top_k, top_p, temp, n_batch,
        repeat_penalty, repeat_penalty_tokens, LLM::globalInstance()->threadCount());
}

void Chat::regenerateResponse()
{
    emit regenerateResponseRequested(); // blocking queued connection
}

void Chat::stopGenerating()
{
    m_llmodel->stopGenerating();
}

QString Chat::response() const
{
    return m_llmodel->response();
}

void Chat::handleResponseChanged()
{
    const int index = m_chatModel->count() - 1;
    m_chatModel->updateValue(index, response());
    emit responseChanged();
}

void Chat::handleModelLoadedChanged()
{
    if (m_shouldDeleteLater)
        deleteLater();
}

void Chat::responseStarted()
{
    m_responseInProgress = true;
    emit responseInProgressChanged();
}

void Chat::responseStopped()
{
    m_responseInProgress = false;
    emit responseInProgressChanged();
    if (m_llmodel->generatedName().isEmpty())
        emit generateNameRequested();
    if (chatModel()->count() < 3)
        Network::globalInstance()->sendChatStarted();
}

QString Chat::modelName() const
{
    return m_llmodel->modelName();
}

void Chat::setModelName(const QString &modelName)
{
    // doesn't block but will unload old model and load new one which the gui can see through changes
    // to the isModelLoaded property
    emit modelNameChangeRequested(modelName);
}

void Chat::newPromptResponsePair(const QString &prompt)
{
    m_chatModel->updateCurrentResponse(m_chatModel->count() - 1, false);
    m_chatModel->appendPrompt(tr("Prompt: "), prompt);
    m_chatModel->appendResponse(tr("Response: "), prompt);
    emit resetResponseRequested(); // blocking queued connection
}

void Chat::serverNewPromptResponsePair(const QString &prompt)
{
    m_chatModel->updateCurrentResponse(m_chatModel->count() - 1, false);
    m_chatModel->appendPrompt(tr("Prompt: "), prompt);
    m_chatModel->appendResponse(tr("Response: "), prompt);
}

bool Chat::isRecalc() const
{
    return m_llmodel->isRecalc();
}

void Chat::loadDefaultModel()
{
    emit loadDefaultModelRequested();
}

void Chat::loadModel(const QString &modelName)
{
    emit loadModelRequested(modelName);
}

void Chat::unloadAndDeleteLater()
{
    if (!isModelLoaded()) {
        deleteLater();
        return;
    }

    m_shouldDeleteLater = true;
    unloadModel();
}

void Chat::unloadModel()
{
    stopGenerating();
    m_llmodel->setShouldBeLoaded(false);
}

void Chat::reloadModel()
{
    m_llmodel->setShouldBeLoaded(true);
}

void Chat::generatedNameChanged()
{
    // Only use the first three words maximum and remove newlines and extra spaces
    QString gen = m_llmodel->generatedName().simplified();
    QStringList words = gen.split(' ', Qt::SkipEmptyParts);
    int wordCount = qMin(3, words.size());
    m_name = words.mid(0, wordCount).join(' ');
    emit nameChanged();
}

void Chat::handleRecalculating()
{
    Network::globalInstance()->sendRecalculatingContext(m_chatModel->count());
    emit recalcChanged();
}

void Chat::handleModelNameChanged()
{
    m_savedModelName = modelName();
    emit modelNameChanged();
}

bool Chat::serialize(QDataStream &stream, int version) const
{
    stream << m_creationDate;
    stream << m_id;
    stream << m_name;
    stream << m_userName;
    stream << m_savedModelName;
    if (!m_llmodel->serialize(stream, version))
        return false;
    if (!m_chatModel->serialize(stream, version))
        return false;
    return stream.status() == QDataStream::Ok;
}

bool Chat::deserialize(QDataStream &stream, int version)
{
    stream >> m_creationDate;
    stream >> m_id;
    emit idChanged();
    stream >> m_name;
    stream >> m_userName;
    emit nameChanged();
    stream >> m_savedModelName;
    // Prior to version 2 gptj models had a bug that fixed the kv_cache to F32 instead of F16 so
    // unfortunately, we cannot deserialize these
    if (version < 2 && m_savedModelName.contains("gpt4all-j"))
        return false;
    m_llmodel->setModelName(m_savedModelName);
    if (!m_llmodel->deserialize(stream, version))
        return false;
    if (!m_chatModel->deserialize(stream, version))
        return false;
    emit chatModelChanged();
    return stream.status() == QDataStream::Ok;
}

QList<QVariantMap> Chat::modelList() const
{
    // Build a model list from exepath and from the localpath
    QList<QVariantMap> list;

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

    QString currentModelName = modelName().isEmpty() ? defaultModel : modelName();

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

    return list;
}

// Helper function to check if a list contains a map with a specific original name
bool Chat::listContainsOriginalName(QList<QVariantMap> list, QString name) const
{
    for (QVariantMap model : list) {
        if (model["original"].toString() == name) {
            return true;
        }
    }
    return false;
}

// Helper function to format model names
QString Chat::formatModelName(QString filename, bool isChatGPT) const
{
    QString name = filename;
    if (!isChatGPT) {
        name.remove(m_regexGGML);
        name.remove(m_regexBinSuffix);
        name.replace('-', ' ');
        name = name.toLower();
        QRegularExpressionMatch match = m_regexWordStart.match(name);
        while (match.hasMatch()) {
            name.replace(match.capturedStart(), 1, match.captured().toUpper());
            match = m_regexWordStart.match(name, match.capturedEnd());
        }
        match = m_regexDigitB.match(name);
        while (match.hasMatch()) {
            name.replace(match.capturedStart(), match.capturedLength(), match.captured(1) + 'B');
            match = m_regexDigitB.match(name, match.capturedEnd());
        }
    }

    name.replace(m_regexGPT4All, "GPT4All");
    name.replace(m_regexGPT, "GPT");
    name.replace(m_regexDoubleGPT, "GPT");

    QRegularExpressionMatch matchQuantization = m_regexQuantization.match(name);
    while (matchQuantization.hasMatch()) {
        QString replacement;
        // If the third captured group (the second digit part after underscore) exists and is not "_0", then keep it in the replacement
        if (!matchQuantization.captured(3).isEmpty() && matchQuantization.captured(3) != "_0") {
            replacement = " " + matchQuantization.captured(2) + "." + matchQuantization.captured(3).mid(1) + "q";
        }
        else {
            replacement = " " + matchQuantization.captured(2) + "q";
        }
        name.replace(matchQuantization.capturedStart(), matchQuantization.capturedLength(), replacement);
        matchQuantization = m_regexQuantization.match(name, matchQuantization.capturedEnd());
    }

    return name;
}

