/**
 * Remotion build configuration.
 *
 * `OffthreadVideo` decodes the ProRes intermediates the conform step produces; the
 * concurrency cap keeps several parallel decoders from saturating memory on a long
 * timeline.
 */
import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("jpeg");
Config.setConcurrency(4);
Config.overrideWebpackConfig((config) => ({
  ...config,
  resolve: {
    ...config.resolve,
    // The craft library lives outside this package (§5), imported by relative path.
    extensions: [".ts", ".tsx", ".js", ".jsx", ...(config.resolve?.extensions ?? [])],
  },
}));
