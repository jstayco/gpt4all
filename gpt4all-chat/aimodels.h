#ifndef AIMODELS_H
#define AIMODELS_H

#include <QObject>
#include <QRegularExpression>
#include <QVariantMap>
#include <QList>
#include <QMap>
#include <mutex>

struct ModelData {
    QString modelName;
    QString modelDisplayName;
    QString source;
    bool isInstalled;
    QVariantMap optionalProperties;
};

class AIModels : public QObject
{
    Q_OBJECT
public:
    static AIModels* globalInstance() {
        static AIModels instance;
        return &instance;
    }

    AIModels(const AIModels&) = delete; // Deleting the copy constructor
    void operator=(const AIModels&) = delete; // Deleting the assignment operator

    Q_INVOKABLE void addModel(const QString& modelName,
                              const QString& modelDisplayName,
                              const QString& source,
                              bool isInstalled,
                              const QVariantMap& optionalProperties = QVariantMap());
    Q_INVOKABLE QVariantMap getModel(const QString& modelName) const;
    Q_INVOKABLE QVariantMap getModel(int index) const;
    Q_INVOKABLE int size() const;
    Q_INVOKABLE QList<QVariantMap> modelList();
    QString getModelName(const QString& modelName) const;
    void updateCurrentModelName(const QString& modelName);

public slots:
    void handleModelNameChanged(const QString& modelName);
    void invalidateModelListCache();

private:
    AIModels(); // Private constructor

    QList<QVariantMap> m_list;
    QMap<QString, ModelData> m_models;
    QRegularExpression m_regexGGML;
    QRegularExpression m_regexBinSuffix;
    QRegularExpression m_regexWordStart;
    QRegularExpression m_regexDigitB;
    QRegularExpression m_regexGPT4All;
    QRegularExpression m_regexGPT;
    QRegularExpression m_regexDoubleGPT;
    QRegularExpression m_regexQuantization;
    QString m_currentModelName;
    mutable std::mutex m_modelListMutex;
    mutable bool m_isModelListInitialized = false;

    QString formatModelName(QString filename, bool isChatGPT) const;
    bool listContainsOriginalName(const QList<QVariantMap>& list, const QString& name) const;

signals:
    void modelListChanged();

};

#endif // AIMODELS_H
