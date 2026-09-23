import edition from './edition.json'

// Release packaging changes this file only for the open-source distribution.
export const OPEN_SOURCE_EDITION = edition.edition === 'opensource'
