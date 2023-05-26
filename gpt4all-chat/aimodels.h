#ifndef AIMODELS_H
#define AIMODELS_H

#include <QObject>
#include <QVariantMap>
#include <QList>
#include <QMap>

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
    explicit AIModels(QObject *parent = nullptr);

    Q_INVOKABLE void addModel(const QString& modelName,
                              const QString& modelDisplayName,
                              const QString& source,
                              bool isInstalled,
                              const QVariantMap& optionalProperties = QVariantMap());
    Q_INVOKABLE QVariantMap getModel(const QString& modelName) const;
    Q_INVOKABLE QVariantMap getModel(int index) const;
    Q_INVOKABLE int size() const;

private:
    QMap<QString, ModelData> m_models;

signals:

};

#endif // AIMODELS_H
