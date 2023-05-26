#include "aimodels.h"

AIModels::AIModels(QObject *parent)
    : QObject{parent}
{

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

int AIModels::size() const {
    return m_models.size();
}
