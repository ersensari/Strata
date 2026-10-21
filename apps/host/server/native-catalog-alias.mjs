/** Recognize the native Swift run config imported from the previous homelab deployment.
 * Keep its public model ID (qwen) stable, while associating it with the download catalog.
 * Do not infer a native model from a quant alone: Poly models can use the same quant.
 */
export function nativeCatalogId(entry, config, catalog) {
 const families=catalog?.families??[];
 if(families.some(f=>(f.sizes??[]).some(s=>s.id===entry.model)))return entry.model;
 if(!/(^|[\\/])strata(?:\.exe)?$/i.test(config.exe??''))return undefined;
 const swift=/^swift-1[._-]5-(.+)$/i.exec(config.model_name??entry.model_name??'');
 if(!swift)return undefined;
 return families.find(f=>f.family==='swift')?.sizes?.find(s=>s.model.toLowerCase()===swift[1].toLowerCase())?.id;
}
